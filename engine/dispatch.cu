#include "kernels.cuh"
#include "attention_fp4_h5_producer.cuh"
#include "attention_fp4_reader.cuh"
#include "attention_fp4_direct_reader.cuh"
#include <stdexcept>
extern "C" void gdn_bits_launch(const h5::GdnArgs&,int,int,bool,cudaStream_t);
extern "C" void gdn_striped_launch(const h5::GdnArgs&,int,bool,cudaStream_t);
namespace h5 {
void qc_original_launch_gdn(const GdnArgs&,int,StateFormat,cudaStream_t);
void qc_original_launch_gdn_single(const GdnArgs&,int,StateFormat,cudaStream_t);
void qc_original_launch_attn_pre(const AttnArgs&,int,cudaStream_t);
void qc_original_launch_attn_tc(const AttnTcArgs&,int,int,int,int,cudaStream_t);
static bool q6(){const char*v=getenv("H5_GDN_BITS");if(!v||!strcmp(v,"8"))return false;if(strcmp(v,"6"))throw std::runtime_error("H5_GDN_BITS needs6 or8");return true;}
static void check_policy(const char*k,const char*v){const char*got=getenv(k);if(got&&strcmp(got,v))throw std::runtime_error(std::string("Q6 requires ")+k+"="+v);}
static void guard(StateFormat f,const GdnArgs&a){
 if(f!=StateFormat::Q8||a.q8_scale!=Q8ScaleMode::LeastSquares)throw std::runtime_error("Q6 needs Q8 allocation + leastsquares scale");
 for(auto kv:{std::pair<const char*,const char*>("H5_GDN_CG","8"),{"H5_GDN_CG_SEGS","0"},{"H5_GDN_ARITH","fp32"},{"H5_GDN_PIPE","0"},{"H5_GDN_SCALED","0"},{"H5_GDN_OCC","2"},{"H5_GDN_SINGLE_PAD_KIB","0"},{"H5_STATE_FUSION","0"},{"H5_MEGA","0"},{"H5_TAIL_PREP","0"},{"H5_AB_VARIANT","0"},{"H5_GDN_VERIFY","off"},{"H5_GDN_SINGLE_DIAG","0"}})check_policy(kv.first,kv.second);
 if(!getenv("H5_NO_SPEC")||strcmp(getenv("H5_NO_SPEC"),"1")||!getenv("H5_GDN_CG"))throw std::runtime_error("Q6 needs explicit no-spec/CG8");
}
static void gdn_dispatch(const GdnArgs&a,int n,StateFormat f,bool single,cudaStream_t st){
 if(!q6()){if(single)qc_original_launch_gdn_single(a,n,f,st);else qc_original_launch_gdn(a,n,f,st);return;}guard(f,a);
 const char*layout=getenv("QC_Q6_LAYOUT");if(layout&&!strcmp(layout,"striped"))gdn_striped_launch(a,n,single,st);else if(!layout||!strcmp(layout,"planes"))gdn_bits_launch(a,n,1,single,st);else throw std::runtime_error("QC_Q6_LAYOUT needsplanes orstriped");
}
void launch_gdn(const GdnArgs&a,int n,StateFormat f,cudaStream_t st){gdn_dispatch(a,n,f,false,st);}
void launch_gdn_single(const GdnArgs&a,int n,StateFormat f,cudaStream_t st){gdn_dispatch(a,n,f,true,st);}
static int kv_mode(){const char*p=getenv("QC_KV");if(!p||!strcmp(p,"fp8"))return 0;if(!strcmp(p,"fp4-staged"))return 1;if(!strcmp(p,"fp4-direct"))return 2;throw std::runtime_error("QC_KV needsfp8,fp4-staged,fp4-direct");}
static int slots(){const char*p=getenv("P12_FP4_SLOTS");int n=p?atoi(p):0;if(n<1||n>256)throw std::runtime_error("P12_FP4_SLOTS must match1..256 engine slots");return n;}
static float* scales(void*p,int ctx){return(float*)((uint8_t*)p+(size_t)slots()*NATTN*NKV*ctx*128);}
void launch_attn_pre(const AttnArgs&a,int rows,cudaStream_t st){
 if(!kv_mode())return qc_original_launch_attn_pre(a,rows,st);
 if(!a.ksc||!a.vsc||a.pf||a.pf_bytes)throw std::runtime_error("FP4 requiresFP8 allocation/no-prefetch");
 auto b=a;b.ksc=scales(a.kc,a.ctx);b.vsc=scales(a.vc,a.ctx);b.vt=0;pdl(p12kv4h5::prepare,dim3(28,rows),dim3(256),0,st,b);
}
void launch_attn_tc(const AttnTcArgs&a,int nqb,int nsp,int rows,int mt,cudaStream_t st){
 int mode=kv_mode();if(!mode||a.hd!=256)return qc_original_launch_attn_tc(a,nqb,nsp,rows,mt,st);
 if(!a.ksc||a.nkv!=4||a.nlayer!=NATTN||a.ring||a.window||a.noncausal||a.draft_block_rows||a.draft_nest)throw std::runtime_error("FP4 target ordinaryattention only");
 auto b=a;b.ksc=scales((void*)a.kc,a.ctx);b.vsc=scales((void*)a.vc,a.ctx);dim3 grid(nqb,4,nsp);
 if(mode==1){
  if(mt<=1)p12kv4tc::attn_launch<1,true,256,6>(b,grid,st);else if(mt==2)p12kv4tc::attn_launch<2,true,256,6>(b,grid,st);else if(mt==3)p12kv4tc::attn_launch<3,true,256,6>(b,grid,st);else throw std::runtime_error("MT>3");
  if(rows>0&&!a.direct)pdl(p12kv4tc::k_attn_combine,dim3(rows,24),dim3(256),0,st,b,nsp,256);
 }else{
  if(mt<=1)p12kv4direct::attn8_launch<1>(b,grid,st);else if(mt==2)p12kv4direct::attn8_launch<2>(b,grid,st);else if(mt==3)p12kv4direct::attn8_launch<3>(b,grid,st);else throw std::runtime_error("MT>3");
  if(rows>0&&!a.direct)pdl(p12kv4direct::k_attn_combine,dim3(rows,24),dim3(256),0,st,b,nsp,256);
 }
}
}
