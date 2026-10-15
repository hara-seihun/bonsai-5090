#include "router-bf16-host.hpp"
#include "gguf.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace halo6000;
void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void reject(const HostImage&h,ErrorCode code){auto r=build_lossless_router_bf16(h);require(!r.ok(),"Invalid image accepted");require(r.error().code==code,"Wrong typed failure");}
int main(int argc,char**argv){try{
 std::vector<uint32_t>bits(RouterBf16HostImage::ImageElements,0x3f800000);
 HostImage h;h.provenance.source_path="synthetic";
 for(int l=0;l<Layers;l++)for(int s=0;s<2;s++){
  HostTensor t;t.name="blk."+std::to_string(l)+(s?".ffn_gate_inp_shexp.weight":".ffn_gate_inp.weight");
  t.shape=s?std::vector<uint64_t>{Hidden}:std::vector<uint64_t>{Hidden,Experts};t.format=WeightFormat::F32;
  t.bytes=(s?Hidden:size_t(Hidden)*Experts)*4;t.data=reinterpret_cast<const uint8_t*>(bits.data()+size_t(l)*RouterBf16HostImage::LayerElements+(s?size_t(Experts)*Hidden:0));h.tensors.push_back(t);
 }
 bits.front()=0x80000000; // Preserve signed zero as bits, not just equality.
 auto valid=build_lossless_router_bf16(h);require(valid.ok(),"Eligible image rejected");require(valid.value().coefficients.front()==0x8000&&valid.value().coefficients.back()==0x3f80,"Packed order/bits wrong");require(valid.value().bytes()==42106880,"Wrong image extent");require(valid.value().provenance.source_path=="synthetic","Provenance lost");
 for(uint32_t bad:{0x3f800001u,0x7f800000u,0xff800000u,0x7fc00000u}){bits.back()=bad;reject(h,ErrorCode::Unsupported);}bits.back()=0x3f800000;
 auto saved=h.tensors.back();h.tensors.pop_back();reject(h,ErrorCode::Geometry);h.tensors.push_back(saved);
 h.tensors.push_back(saved);reject(h,ErrorCode::Geometry);h.tensors.pop_back();
 h.tensors.back().shape={Hidden,1};reject(h,ErrorCode::Geometry);h.tensors.back()=saved;
 h.tensors.back().bytes-=4;reject(h,ErrorCode::Geometry);h.tensors.back()=saved;
 h.tensors.back().data=nullptr;reject(h,ErrorCode::Geometry);h.tensors.back()=saved;
 h.tensors.back().format=WeightFormat::Q8Grouped128;reject(h,ErrorCode::Unsupported);h.tensors.back()=saved;
 if(argc==3){
  auto g=Gguf::open(argv[1]);require(g.ok(),"Cannot load source GGUF");HostImage actual;actual.provenance.source=g.value().storage->identity;actual.provenance.source_path=argv[1];actual.storage=g.value().storage;
  for(const auto&t:g.value().tensors)if(t.name.find(".ffn_gate_inp.weight")!=std::string::npos||t.name.find(".ffn_gate_inp_shexp.weight")!=std::string::npos){require(t.type==0,"Source router not F32");HostTensor v;v.name=t.name;v.shape=t.shape;v.data=t.data;v.bytes=t.bytes;v.format=WeightFormat::F32;actual.tensors.push_back(v);}
  auto result=build_lossless_router_bf16(actual);require(result.ok(),"Actual source ineligible");auto&image=result.value().coefficients;
  std::vector<uint16_t>prior(image.size());std::ifstream f(argv[2],std::ios::binary);f.read(reinterpret_cast<char*>(prior.data()),prior.size()*2);require(bool(f)&&f.peek()==EOF,"Guarded image extent mismatch");require(prior==image,"Production builder != guarded image");
  std::cout<<"actual_source_coefficients_verified="<<image.size()<<" production_image_bytes="<<image.size()*2<<"\n";
 }else require(argc==1,"Usage: router-bf16-host-test [MODEL GUARDED_IMAGE]");
 std::cout<<"positive_images=1 invalid_images_rejected=10 signed_zero_preserved=true\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
