// Real PTQ1 scale statistics and exact dense5 component fixture export. No CUDA required.
#include "gguf.h"
#include "compact.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>
static double half_value(uint16_t u) {
 const int e=(u>>10)&31,m=u&1023; if(e==31)return m?NAN:INFINITY;
 return (u&32768?-1.:1.)*std::ldexp(e?1024.+m:double(m),e?e-25:-24);
}
static double scale(const uint8_t* p) { uint16_t u;memcpy(&u,p+26,2);double v=half_value(u);if(!std::isfinite(v)||v<0)throw std::runtime_error("invalid weight scale");return v; }
static void save(const std::filesystem::path& path,const void* p,size_t n) {
 FILE*f=fopen(path.c_str(),"wbx");if(!f)throw std::runtime_error("output exists or inaccessible: "+path.string());
 if(fwrite(p,1,n,f)!=n||fclose(f))throw std::runtime_error("output write failed");
}
static double percentile(std::vector<double>&v,double q){std::sort(v.begin(),v.end());return v[size_t(q*(v.size()-1))];}
int main(int argc,char**argv){try{
 if(argc!=4 && argc!=5){fprintf(stderr,"usage: scale-carrier-model stats MODEL NEW_JSONL | gu MODEL LAYER NEW_DIR | tensor MODEL TENSOR_NAME NEW_DIR\n");return 2;}
 h5::Gguf gg;gg.open(argv[2]);std::string mode=argv[1];
 if(mode=="stats" && argc==4){
  FILE*f=fopen(argv[3],"wx");if(!f)throw std::runtime_error("stats output unavailable");
  for(const auto&t:gg.tensors)if(t.type==143&&t.ne.size()==2){
   size_t K=t.ne[0],N=t.ne[1],B=K/128;if(!B)throw std::runtime_error("empty contraction");std::vector<double>cv,range;size_t zero=0;
   double overall=0,err2=0,magnitude=0;
   for(size_t row=0;row<N;row++){
    double sum=0,sq=0,lo=INFINITY,hi=0;
    for(size_t b=0;b<B;b++){double v=scale(t.data+(row*B+b)*28);sum+=v;sq+=v*v;lo=std::min(lo,v);hi=std::max(hi,v);}
    double mu=sum/B,var=std::max(0.,sq/B-mu*mu);if(mu==0)zero++;cv.push_back(mu?sqrt(var)/mu:0);range.push_back(mu?(hi-lo)/mu:0);overall+=mu;err2+=var*B;magnitude+=sq;
   }
   fprintf(f,"{\"tensor\":\"%s\",\"N\":%zu,\"K\":%zu,\"row_mean_scale\":%.9g,\"zero_rows\":%zu,\"scale_rel_rms\":%.9g,\"cv_p50\":%.9g,\"cv_p90\":%.9g,\"cv_max\":%.9g,\"range_over_mean_p50\":%.9g,\"range_over_mean_p90\":%.9g}\n",t.name.c_str(),N,K,overall/N,zero,magnitude?sqrt(err2/magnitude):0,percentile(cv,.5),percentile(cv,.9),percentile(cv,1.),percentile(range,.5),percentile(range,.9));
  }if(ferror(f)||fclose(f))throw std::runtime_error("stats write failed");return 0;
 }
 if(argc!=5||(mode!="gu"&&mode!="tensor"))throw std::runtime_error("invalid export mode");
 const h5::GTensor *a,*b=nullptr;
 if(mode=="gu"){char*end;long l=strtol(argv[3],&end,10);if(*end||l<0||l>=64)throw std::runtime_error("bad layer");a=&gg.t("blk."+std::to_string(l)+".ffn_gate.weight");b=&gg.t("blk."+std::to_string(l)+".ffn_up.weight");}
 else a=&gg.t(argv[3]);
 if(a->type!=143||a->ne.size()!=2||(b&&(b->type!=143||b->ne!=a->ne)))throw std::runtime_error("not a matching PTQ1 matrix");
 int K=a->ne[0],Nr=a->ne[1],N=b?2*Nr:Nr,B=K/128;if(K%128||N%32)throw std::runtime_error("unsupported geometry");
 std::vector<uint8_t>W(size_t(N/32)*B*832);std::vector<uint16_t>S(size_t(N)*B);
 for(int g=0;g<N/32;g++)for(int kb=0;kb<B;kb++){
  int8_t trit[32][128];uint16_t sh[32];const auto&t=b?(g%2?*b:*a):*a;int group=b?g/2:g;
  for(int r=0;r<32;r++){const uint8_t*p=t.data+(size_t(group*32+r)*B+kb)*28;scale(p);dense5::decode_ptq(p,trit[r],sh[r]);}
  uint8_t*dst=W.data()+(size_t(kb)*(N/32)+g)*832;
  for(int lane=0;lane<32;lane++){
   int8_t stream[128];int gid=lane>>2,tq=lane&3;
   for(int w=0;w<8;w++){int m=w>>2,h=(w>>1)&1,r=w&1,row=16*m+8*r+gid;for(int j=0;j<8;j++){stream[w*16+j]=trit[row][64*h+8*tq+j];stream[w*16+8+j]=trit[row][64*h+32+8*tq+j];}}
   dense5::pack_lane(dst,lane,stream);
  }
  for(int gid=0;gid<8;gid++)for(int q=0;q<4;q++)S[(size_t(kb)*(N/32)+g)*32+gid*4+q]=sh[16*(q>>1)+8*(q&1)+gid];
 }
 std::filesystem::path dir=argv[4];if(std::filesystem::exists(dir))throw std::runtime_error("fixture directory exists");std::filesystem::create_directories(dir);
 save(dir/"weights.bin",W.data(),W.size());save(dir/"scales.bin",S.data(),S.size()*2);
 std::string metadata="{\"N\":"+std::to_string(N)+",\"K\":"+std::to_string(K)+",\"fused\":"+(b?"true":"false")+",\"tensor\":\""+a->name+"\"}\n";save(dir/"fixture.json",metadata.data(),metadata.size());
 puts(metadata.c_str());return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
