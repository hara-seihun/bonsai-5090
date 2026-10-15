#include "decode.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
using namespace halo6000;
static bool ok(Status s){if(s.ok())return true;fprintf(stderr,"%s\n",s.error().message.c_str());return false;}
static std::vector<float>read(const float*p,size_t n){std::vector<float>x(n);if(!ok(cuda_status(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost),"schedule copy")))exit(2);return x;}
static size_t errors(const std::vector<float>&a,const std::vector<float>&b){size_t n=0;for(size_t i=0;i<a.size();i++)n+=memcmp(&a[i],&b[i],4)!=0;return n;}
int main(int argc,char**argv){
 if(argc!=3){fprintf(stderr,"schedule MODEL CACHE_DIR\n");return 2;}
 auto image=load_cached(argv[1],argv[2]);if(!image.ok()){fprintf(stderr,"%s\n",image.error().message.c_str());return 2;}
 auto created=DeviceModel::create(image.value(),1,128);if(!created.ok()){fprintf(stderr,"%s\n",created.error().message.c_str());return 2;}auto m=std::move(created.value());
 auto made=Workspace::create(128);if(!made.ok())return 2;auto w=std::move(made.value());
 // Flags-off must be bit-identical even in the larger compiled rotation kernel.
 std::vector<std::vector<float>>reference;
 for(int variant=0;variant<2;variant++){
  m->has_rotation=variant;if(!ok(m->reset()))return 2;size_t bad=0;
  for(int p=0;p<4;p++){
   if(!ok(run_token(*m,w->view,100+p*31,p,0,188)))return 2;
   auto logits=read(w->view.logits,Vocab);auto state=read(m->state.gdn,GdnFloatsPerSlot);
   if(!variant){reference.push_back(std::move(logits));reference.push_back(std::move(state));}
   else bad+=errors(reference[2*p],logits)+errors(reference[2*p+1],state);
  }
  if(variant){printf("{\"guard\":\"flags_off_two_compiled_variants\",\"steps\":4,\"bit_errors\":%zu}\n",bad);if(bad)return 1;}
 }
 for(int lanes:{16,8}){
  m->matvec_lanes=lanes;m->has_rotation=false;if(!ok(m->reset()))return 2;size_t bad=0;
  for(int p=0;p<4;p++){
   if(!ok(run_token(*m,w->view,100+p*31,p,0,188)))return 2;
   bad+=errors(reference[2*p],read(w->view.logits,Vocab))+errors(reference[2*p+1],read(m->state.gdn,GdnFloatsPerSlot));
  }
  printf("{\"guard\":\"whole_token_exact_row_mapping\",\"row_lanes\":%d,\"steps\":4,\"bit_errors\":%zu}\n",lanes,bad);if(bad)return 1;
 }
 m->matvec_lanes=32;reference.clear();
 // Deliberately transformed activation inputs, unchanged weights. This checks
 // epoch placement, not quality of any rotated quantized model.
 struct Frame{std::vector<float>hidden,state,history;};
 const LayerWeights original=m->weights.layer[0];
 for(int b:{128,256,512,4096}){
  auto&l=m->weights.layer[0];l=original;l.output.flags=rotation_flags(b==4096?12:b==512?9:b==256?8:7);
  const int down=b>512?512:b;l.expert_down.flags=l.shared_down.flags=rotation_flags(down==512?9:down==256?8:7);
  m->has_rotation=true;if(!ok(cuda_status(cudaMemcpy(m->device_weights,&m->weights,sizeof m->weights,cudaMemcpyHostToDevice),"schedule descriptor")))return 2;
  std::vector<Frame>frames;size_t bad=0;
  for(int path=0;path<2;path++){
   if(!ok(m->reset()))return 2;
   for(int p=0;p<4;p++){
    if(!ok(embed(*m,w->view,100+p*31)))return 2;
    if(path){if(!ok(run_gdn_layer(*m,w->view,0,0,188)))return 2;}
    else for(int phase=0;phase<=int(LayerPhase::Down);phase++)if(!ok(run_gdn_phase(*m,w->view,0,0,LayerPhase(phase),188)))return 2;
    Frame got{read(w->view.hidden,Hidden),read(m->state.gdn,ValueHeads*ValueDim*StateDim),read(m->state.conv,(ConvTaps-1)*ConvChannels)};
    if(!path)frames.push_back(std::move(got));else bad+=errors(frames[p].hidden,got.hidden)+errors(frames[p].state,got.state)+errors(frames[p].history,got.history);
   }
  }
  printf("{\"guard\":\"rotated_input_control_vs_persistent\",\"writer_block\":%d,\"down_block\":%d,\"steps\":4,\"bit_errors\":%zu,\"quality_claim\":false}\n",b,down,bad);if(bad)return 1;
 }
 return 0;
}
