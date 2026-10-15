#include "kernels.cuh"
#include "gdn_single.h"
#include "q8_scale.h"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace h5 {
void launch_gdn(const GdnArgs&, int, StateFormat, cudaStream_t);
void launch_conv(const ConvArgs&, int, StateFormat, cudaStream_t);
}
using namespace h5;
#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e)); std::exit(2); } } while(0)
static void require(bool ok,const char* label) { if(!ok) { fprintf(stderr,"contract failed: %s\n",label); std::exit(1); } }
constexpr size_t M=(size_t)HV*128*128;
constexpr int TEST_SLOTS=4, ACTIVE=3, ROWS=21;
static const char* scale_name() { return q8_scale_name(gdn_q8_scale_mode()); }
static void json(const char* format,...) {
  require(format[0]=='{',"JSON record begins with object");
  printf("{\"q8_scale\":\"%s\",",scale_name());
  va_list args; va_start(args,format); vprintf(format+1,args); va_end(args);
}
static void paired_json(const char* format,...) {
  require(format[0]=='{',"JSON record begins with object");
  printf("{\"q8_scale\":\"max_vs_ls\",");
  va_list args; va_start(args,format); vprintf(format+1,args); va_end(args);
}
struct ScaleEnvironment {
  bool present=getenv("H5_GDN_Q8_SCALE")!=nullptr;
  std::string value=present?getenv("H5_GDN_Q8_SCALE"):"";
  ~ScaleEnvironment() { require((present?setenv("H5_GDN_Q8_SCALE",value.c_str(),1):unsetenv("H5_GDN_Q8_SCALE"))==0,"restore Q8 scale environment"); }
};
static void set_scale(const char* value) {
  require((value?setenv("H5_GDN_Q8_SCALE",value,1):unsetenv("H5_GDN_Q8_SCALE"))==0,"set Q8 scale environment");
}
static void selector_contracts() {
  ScaleEnvironment restore;
  set_scale(nullptr); require(gdn_q8_scale_mode()==Q8ScaleMode::LeastSquares,"unset Q8 scale explicitly selects least squares");
  for(const char* value:{"max","ls","max","ls"}) {
    set_scale(value);
    require(gdn_q8_scale_mode()==(!strcmp(value,"max")?Q8ScaleMode::Max:Q8ScaleMode::LeastSquares),"selector rereads each arm");
    json("{\"kind\":\"contract\",\"operation\":\"scale_selector\",\"status\":\"pass\"}\n");
  }
  for(const char* value:{"","LS","least-squares","unknown"}) {
    set_scale(value); bool rejected=false;
    try { (void)gdn_q8_scale_mode(); } catch(const std::exception&) { rejected=true; }
    require(rejected,"unknown or empty Q8 scale is rejected before launch");
  }
  validate_q8_scale_mode(Q8ScaleMode::Max); validate_q8_scale_mode(Q8ScaleMode::LeastSquares);
  bool invalid_enum=false;
  try { validate_q8_scale_mode(static_cast<Q8ScaleMode>(-1)); } catch(const std::exception&) { invalid_enum=true; }
  require(invalid_enum,"invalid explicit Q8 scale enum is rejected");
  set_scale(nullptr);
  json("{\"kind\":\"contract\",\"operation\":\"scale_selector_unset_and_invalid\",\"status\":\"pass\"}\n");
}
static const char* name(StateFormat f) {
  switch(f) { case StateFormat::Q8: return "q8"; case StateFormat::F16: return "f16"; case StateFormat::F32: return "f32"; case StateFormat::BF16: return "bf16"; }
  std::abort();
}
static size_t layer_bytes(StateFormat f) { return gdn_state_slot_bytes(f)/NGDN; }
struct Buffer {
  void* p=nullptr; size_t bytes;
  explicit Buffer(size_t n):bytes(n) { CU(cudaMalloc(&p,n)); CU(cudaMemset(p,0,n)); CU(cudaStreamSynchronize(nullptr)); }
  ~Buffer() { if(p) CU(cudaFree(p)); }
  Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
  template<class T> T* as() const { return static_cast<T*>(p); }
  void put(const void* src,size_t n,size_t off=0) { require(off+n<=bytes,"buffer write range"); CU(cudaMemcpy((char*)p+off,src,n,cudaMemcpyHostToDevice)); CU(cudaStreamSynchronize(nullptr)); }
  template<class T> std::vector<T> get(size_t n,size_t off=0) const { require(off+n*sizeof(T)<=bytes,"buffer read range"); std::vector<T> v(n); CU(cudaMemcpy(v.data(),(char*)p+off,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
};
static std::vector<float> seed(int slot) {
  std::vector<float> v(M);
  for(size_t i=0;i<M;i++) v[i]=.03f*sinf(float(i%8191)*.17f+slot*.73f);
  return v;
}
static GdnQ8Layer encode(const std::vector<float>& v) {
  const Q8ScaleMode mode=gdn_q8_scale_mode(); GdnQ8Layer q{};
  for(int h=0;h<HV;h++) for(int r=0;r<128;r++) {
    const size_t base=((size_t)h*128+r)*128; float amax=0;
    for(int c=0;c<128;c++) amax=std::max(amax,fabsf(v[base+c]));
    const float scale=amax*(1.f/127.f),inv=scale>0?1.f/scale:0.f;
    double numerator=0; int denominator=0;
    for(int c=0;c<128;c++) {
      const int code=std::max(-127,std::min(127,(int)lrintf(v[base+c]*inv)));
      q.data[h][r][c]=code; numerator+=(double)v[base+c]*code; denominator+=code*code;
    }
    switch(mode) {
      case Q8ScaleMode::Max: q.scales[h][r]=scale; break;
      case Q8ScaleMode::LeastSquares: q.scales[h][r]=denominator?(float)(numerator/denominator):0.f; break;
    }
  }
  return q;
}
static void seed_layer(Buffer& b,StateFormat f,int slots,int gl) {
  for(int s=0;s<slots;s++) {
    const auto v=seed(s); const size_t off=(s*(size_t)NGDN+gl)*layer_bytes(f);
    if(f==StateFormat::Q8) { const auto q=encode(v); b.put(&q,sizeof(q),off); }
    else if(f==StateFormat::F16) { std::vector<__half> x(M); for(size_t i=0;i<M;i++) x[i]=__float2half(v[i]); b.put(x.data(),M*sizeof(__half),off); }
    else b.put(v.data(),M*sizeof(float),off);
  }
}
static std::vector<float> decode_layer(const Buffer& b,StateFormat f,int slot,int gl) {
  const size_t off=(slot*(size_t)NGDN+gl)*layer_bytes(f); std::vector<float> v(M);
  if(f==StateFormat::Q8) {
    const auto q=b.get<GdnQ8Layer>(1,off);
    for(int h=0;h<HV;h++) for(int r=0;r<128;r++) {
      const float scale=q[0].scales[h][r]; require(std::isfinite(scale)&&scale>=0,"finite nonnegative Q8 scale");
      for(int c=0;c<128;c++) { const int x=q[0].data[h][r][c]; require(x>=-127,"Q8 excludes -128"); v[((size_t)h*128+r)*128+c]=x*scale; }
    }
  } else if(f==StateFormat::F16) { const auto x=b.get<__half>(M,off); for(size_t i=0;i<M;i++) v[i]=__half2float(x[i]); }
  else v=b.get<float>(M,off);
  return v;
}
static void metric(const char* label,const char* format,const std::vector<float>& a,const std::vector<float>& b,int step) {
  require(a.size()==b.size(),"metric dimensions"); double ss=0,ref=0,maxabs=0; size_t nonfinite=0;
  for(size_t i=0;i<a.size();i++) {
    if(!std::isfinite(a[i])||!std::isfinite(b[i])) { nonfinite++; continue; }
    const double d=(double)b[i]-a[i]; ss+=d*d; ref+=(double)a[i]*a[i]; maxabs=std::max(maxabs,fabs(d));
  }
  json("{\"kind\":\"quality\",\"tensor\":\"%s\",\"format\":\"%s\",\"step\":%d,\"count\":%zu,\"nonfinite_pairs\":%zu,\"max_abs\":%.9g,\"rms_abs\":%.9g,\"relative_l2\":",label,format,step,a.size(),nonfinite,maxabs,sqrt(ss/std::max(size_t(1),a.size()-nonfinite)));
  if(ref>0) printf("%.9g",sqrt(ss/ref)); else printf("null"); puts("}");
}
struct Inputs {
  Buffer conv{(size_t)ACTIVE*ROWS*CONV_CH*sizeof(float)}, gates{(size_t)ACTIVE*ROWS*2*HV*sizeof(float)}, segs{ACTIVE*sizeof(Seg)};
  Inputs() {
    std::vector<float> co((size_t)ACTIVE*ROWS*CONV_CH), gs((size_t)ACTIVE*ROWS*2*HV);
    for(int r=0;r<ACTIVE*ROWS;r++) {
      for(int qk=0;qk<2;qk++) for(int h=0;h<HK;h++) {
        const size_t base=(size_t)r*CONV_CH+qk*KDIM+h*128; float norm=0;
        for(int c=0;c<128;c++) { const float x=sinf(r*.19f+c*.31f+h*.13f+qk*.7f); co[base+c]=x; norm+=x*x; }
        for(int c=0;c<128;c++) co[base+c]/=sqrtf(norm);
      }
      for(int c=0;c<VDIM;c++) co[(size_t)r*CONV_CH+2*KDIM+c]=.5f*sinf(c*.11f+r*.17f);
      for(int h=0;h<HV;h++) { gs[(size_t)r*2*HV+h]=.93f+.065f*(h%7)/6.f; gs[(size_t)r*2*HV+HV+h]=.2f+.7f*(h%5)/4.f; }
    }
    conv.put(co.data(),co.size()*sizeof(float)); gates.put(gs.data(),gs.size()*sizeof(float));
  }
  void single_step(int step) {
    std::vector<float> co(CONV_CH),gs(2*HV); std::vector<Seg> seg(ACTIVE);
    const int slots[ACTIVE]={step<=32?0:3,1,2};
    for(int s=0;s<ACTIVE;s++) {
      for(int qk=0;qk<2;qk++) for(int h=0;h<HK;h++) {
        const size_t base=qk*KDIM+h*128; float norm=0;
        for(int c=0;c<128;c++) { const float x=sinf(step*.137f+s*.29f+c*.31f+h*.13f+qk*.7f); co[base+c]=x; norm+=x*x; }
        for(int c=0;c<128;c++) co[base+c]/=sqrtf(norm);
      }
      for(int c=0;c<VDIM;c++) co[2*KDIM+c]=.5f*sinf(c*.11f+step*.17f+s*.43f);
      for(int h=0;h<HV;h++) { gs[h]=.93f+.065f*((h+step+s)%7)/6.f; gs[HV+h]=.2f+.7f*((h+3*step+s)%5)/4.f; }
      conv.put(co.data(),co.size()*sizeof(float),(size_t)s*ROWS*CONV_CH*sizeof(float));
      gates.put(gs.data(),gs.size()*sizeof(float),(size_t)s*ROWS*2*HV*sizeof(float));
      const int at=step%2?(s+step)%ACTIVE:(ACTIVE-1-s+step)%ACTIVE;
      seg[at]={slots[s],s*ROWS,1,0,0,step%2};
      require(gdn_single_eligible(seg[at]),"ordinary single-token segment");
    }
    segs.put(seg.data(),seg.size()*sizeof(Seg));
  }
  void segments(int len,int commit,int keep,int par) {
    std::vector<Seg> v(ACTIVE);
    for(int s=0;s<ACTIVE;s++) v[s]={(s+2)%ACTIVE,s*ROWS,len,commit,keep,par};
    segs.put(v.data(),v.size()*sizeof(Seg));
  }
};
struct State {
  StateFormat format;
  Buffer committed,pending,y;
  State(StateFormat f,int slots):format(f),committed(slots*gdn_state_slot_bytes(f)),
    pending((size_t)slots*NGDN*2*PMAX*PEND_GDN*pending_state_element_bytes(f)),y((size_t)ACTIVE*ROWS*VDIM*sizeof(float)) {}
  GdnArgs args(const Inputs& in,int gl,bool delta) {
    GdnArgs a{}; a.conv_out=in.conv.as<float>(); a.gates=in.gates.as<float>(); a.state=committed.p; a.pend=pending.p; a.y=y.as<float>(); a.segs=in.segs.as<Seg>(); a.gl=gl; a.replay_delta=delta; a.q8_scale=gdn_q8_scale_mode(); return a;
  }
};
static void scale_contracts(cudaStream_t st) {
  ScaleEnvironment restore; Inputs in;
  State maximum(StateFormat::Q8,1),least_squares(StateFormat::Q8,1);
  constexpr int gl=1;
  const Seg segment{0,0,1,0,0,0}; in.segs.put(&segment,sizeof(segment));
  for(bool single:{false,true}) for(float amplitude:{0.f,1.f,1e13f}) {
    std::vector<float> co(CONV_CH,0.f),gates(2*HV,0.f),values(M);
    for(int h=0;h<HK;h++) for(int c=0;c<128;c++)
      co[KDIM+h*128+c]=c%11==0?0.f:(float(int((c*37+h*19)%251)-125)/125.f);
    for(int h=0;h<HV;h++) {
      gates[HV+h]=1.f;
      for(int r=0;r<128;r++) {
        const float v=r%9==0?0.f:amplitude*(.25f+.005f*r)*(r%2?-1.f:1.f);
        co[2*KDIM+h*128+r]=v;
        for(int c=0;c<128;c++) values[((size_t)h*128+r)*128+c]=v*co[KDIM+(h%HK)*128+c];
      }
    }
    in.conv.put(co.data(),co.size()*sizeof(float)); in.gates.put(gates.data(),gates.size()*sizeof(float));
    for(State* state:{&maximum,&least_squares}) {
      CU(cudaMemsetAsync(state->committed.p,0,state->committed.bytes,st));
      set_scale(state==&maximum?"max":"ls");
      const GdnArgs args=state->args(in,gl,false);
      set_scale(state==&maximum?"ls":"max");
      if(single) launch_gdn_single(args,1,StateFormat::Q8,st);
      else launch_gdn(args,1,StateFormat::Q8,st);
      CU(cudaGetLastError()); CU(cudaStreamSynchronize(st));
    }
    const size_t offset=(size_t)gl*sizeof(GdnQ8Layer);
    const auto maximum_layer=maximum.committed.get<GdnQ8Layer>(1,offset),ls_layer=least_squares.committed.get<GdnQ8Layer>(1,offset);
    const auto& maxq=maximum_layer[0]; const auto& lsq=ls_layer[0];
    require(!memcmp(maxq.data,lsq.data,M),"max and LS saves have identical packed codes from identical state/inputs");
    set_scale("max"); const auto reference=encode(values);
    require(!memcmp(maxq.data,reference.data,M),"known GDN rows retain max RTN signed codes");
    double worst_relative=0; int zero_rows=0,different_scales=0,max_denominator=0;
    for(int h=0;h<HV;h++) for(int r=0;r<128;r++) {
      double numerator=0; int denominator=0; const size_t base=((size_t)h*128+r)*128;
      for(int c=0;c<128;c++) {
        const int code=lsq.data[h][r][c]; require(code>=-127,"LS code excludes -128");
        numerator+=(double)values[base+c]*code; denominator+=code*code;
      }
      require(denominator<=2064512,"exact INT32 row denominator bound");
      max_denominator=std::max(max_denominator,denominator);
      const double fitted=denominator?numerator/denominator:0.;
      const float scale=lsq.scales[h][r],max_scale=maxq.scales[h][r];
      require(std::isfinite(scale)&&scale>=0&&std::isfinite(max_scale)&&max_scale>=0,"finite nonnegative scales including real 1e13 range");
      require(max_scale==reference.scales[h][r],"max row scale unchanged");
      if(!denominator) {
        require(scale==0.f&&!std::signbit(scale)&&max_scale==0.f&&!std::signbit(max_scale),"zero denominator produces exact positive zero scale");
        zero_rows++;
      } else {
        const double relative=fabs(scale-fitted)/fitted; worst_relative=std::max(worst_relative,relative);
        require(relative<=4e-6,"LS FP32 fitted scale agrees with independent FP64 host reference");
        different_scales+=scale!=max_scale;
      }
    }
    require(amplitude==0||different_scales>0,"LS arm changes stored scale on nonzero fixture");
    if(amplitude==0) for(State* state:{&maximum,&least_squares}) {
      const auto bytes=state->committed.get<unsigned char>(state->committed.bytes);
      for(unsigned char byte:bytes) require(byte==0,"zero GDN state including all scales stays exact zero");
    }
    for(State* state:{&maximum,&least_squares}) {
      set_scale(state==&maximum?"max":"ls");
      const auto output=state->y.get<float>(VDIM);
      for(float value:output) require(std::isfinite(value),"known-row output remains finite");
      json("{\"kind\":\"contract\",\"operation\":\"row_scale\",\"route\":\"%s\",\"input_amplitude\":%.9g,\"identical_packed_codes\":true,\"explicit_mode_ignores_environment\":true,\"zero_rows\":%d,\"different_scales\":%d,\"max_denominator\":%d,\"ls_reference_max_relative_error\":%.9g,\"status\":\"pass\"}\n",single?"single_q8":"generic_q8",amplitude,zero_rows,different_scales,max_denominator,worst_relative);
    }
  }
}
static std::vector<float> active_output(const State& s,int len) {
  std::vector<float> out;
  for(int r=0;r<ACTIVE;r++) { const auto v=s.y.get<float>((size_t)len*VDIM,(size_t)r*ROWS*VDIM*sizeof(float)); out.insert(out.end(),v.begin(),v.end()); }
  return out;
}
static void quality(cudaStream_t st) {
  Inputs in; State f32(StateFormat::F32,TEST_SLOTS),f16(StateFormat::F16,TEST_SLOTS),q8(StateFormat::Q8,TEST_SLOTS);
  const int steps[][4]={{8,0,1,0},{8,0,1,1},{8,3,1,0},{21,8,0,1}};
  for(bool delta:{false,true}) for(int gl:{1,NGDN-1}) {
    for(State* s:{&f32,&f16,&q8}) { CU(cudaMemset(s->committed.p,0,s->committed.bytes)); CU(cudaMemset(s->pending.p,0,s->pending.bytes)); seed_layer(s->committed,s->format,TEST_SLOTS,gl); }
    for(int step=0;step<4;step++) {
      in.segments(steps[step][0],steps[step][1],steps[step][2],steps[step][3]);
      for(State* s:{&f32,&f16,&q8}) launch_gdn(s->args(in,gl,delta),ACTIVE,s->format,st);
      CU(cudaGetLastError()); CU(cudaStreamSynchronize(st));
      json("{\"kind\":\"case\",\"step\":%d,\"layer\":%d,\"replay_delta\":%s,\"new_rows\":%d,\"accept\":%d,\"keep\":%d}\n",step,gl,delta?"true":"false",steps[step][0],steps[step][1],steps[step][2]);
      const auto y=active_output(f32,steps[step][0]);
      for(State* s:{&f16,&q8}) {
        metric("output",name(s->format),y,active_output(*s,steps[step][0]),step);
        for(int slot=0;slot<ACTIVE;slot++) metric("committed_state",name(s->format),decode_layer(f32.committed,StateFormat::F32,slot,gl),decode_layer(s->committed,s->format,slot,gl),step);
      }
    }
  }
}
static void bitwise(const char* tensor,const Buffer& a,const Buffer& b,size_t bytes,size_t offset,int step,int layer,int slot) {
  const auto av=a.get<unsigned char>(bytes,offset),bv=b.get<unsigned char>(bytes,offset);
  size_t different=0,first=bytes;
  for(size_t i=0;i<bytes;i++) if(av[i]!=bv[i]) { different++; first=std::min(first,i); }
  json("{\"kind\":\"equivalence\",\"control_route\":\"generic_q8\",\"candidate_route\":\"single_q8\",\"tensor\":\"%s\",\"step\":%d,\"layer\":%d,\"slot\":%d,\"bytes\":%zu,\"different_bytes\":%zu,\"first_different_byte\":",tensor,step,layer,slot,bytes,different);
  if(different) printf("%zu",first); else printf("null");
  printf(",\"status\":\"%s\"}\n",different?"fail":"pass");
  require(different==0,"generic Q8 and separate single Q8 are bit-equivalent");
}
static void single_contracts(cudaStream_t st) {
  Inputs in; State generic(StateFormat::Q8,TEST_SLOTS),single(StateFormat::Q8,TEST_SLOTS);
  const int layers[]={0,1,NGDN-1};
  const size_t sb=gdn_state_slot_bytes(StateFormat::Q8);
  for(int gl:layers) seed_layer(generic.committed,StateFormat::Q8,ACTIVE,gl);
  CU(cudaMemcpyAsync(single.committed.p,generic.committed.p,generic.committed.bytes,cudaMemcpyDeviceToDevice,st));
  CU(cudaStreamSynchronize(st));
  for(int step=1;step<=96;step++) {
    in.single_step(step);
    const bool checkpoint=step==1||step%16==0||step==33||step==65||step==96;
    for(int gl:layers) {
      launch_gdn(generic.args(in,gl,false),ACTIVE,StateFormat::Q8,st);
      launch_gdn_single(single.args(in,gl,false),ACTIVE,StateFormat::Q8,st);
      CU(cudaGetLastError());
      if(checkpoint) {
        CU(cudaStreamSynchronize(st));
        for(int row=0;row<ACTIVE;row++) bitwise("output",generic.y,single.y,VDIM*sizeof(float),(size_t)row*ROWS*VDIM*sizeof(float),step,gl,row==0&&step>32?3:row);
        for(int slot=0;slot<TEST_SLOTS;slot++) {
          const size_t off=(slot*(size_t)NGDN+gl)*sizeof(GdnQ8Layer);
          bitwise("state_payload",generic.committed,single.committed,M,off,step,gl,slot);
          bitwise("state_scales",generic.committed,single.committed,HV*128*sizeof(float),off+M,step,gl,slot);
        }
        metric("output","single_q8_vs_generic_q8",active_output(generic,1),active_output(single,1),step);
      }
    }
    // Host writes for the next step must not race the nonblocking recurrence stream.
    CU(cudaStreamSynchronize(st));
    if(step==32||step==64) {
      for(State* s:{&generic,&single}) {
        const auto source=s->committed.get<unsigned char>(sb);
        if(step==32) CU(cudaMemcpyAsync((char*)s->committed.p+3*sb,s->committed.p,sb,cudaMemcpyDeviceToDevice,st));
        else CU(cudaMemsetAsync((char*)s->committed.p+3*sb,0,sb,st));
        CU(cudaStreamSynchronize(st));
        const auto target=s->committed.get<unsigned char>(sb,3*sb);
        if(step==32) require(source==target,"clone copies entire Q8 slot including scales");
        else for(auto x:target) require(x==0,"reset clears entire reused Q8 slot including scales");
        require(source==s->committed.get<unsigned char>(sb),"clone/reset leaves source slot unchanged");
      }
      json("{\"kind\":\"contract\",\"route\":\"generic_q8_vs_single_q8\",\"operation\":\"%s\",\"step\":%d,\"source_slot\":0,\"reused_slot\":3,\"slot_bytes_including_scales\":%zu,\"status\":\"pass\"}\n",step==32?"clone":"reset",step,sb);
    }
  }
  for(int slot=0;slot<TEST_SLOTS;slot++) for(int gl=0;gl<NGDN;gl++) {
    const size_t off=(slot*(size_t)NGDN+gl)*sizeof(GdnQ8Layer);
    bitwise("final_state_including_scales",generic.committed,single.committed,sizeof(GdnQ8Layer),off,96,gl,slot);
    if(gl!=0&&gl!=1&&gl!=NGDN-1) {
      const auto bytes=generic.committed.get<unsigned char>(sizeof(GdnQ8Layer),off);
      for(auto x:bytes) require(x==0,"ordinary single trajectory leaves other layers untouched");
    }
  }
}
static std::vector<float> replay_reference(std::vector<float> m,const std::vector<float>& p,int accepted,bool delta) {
  for(int h=0;h<HV;h++) for(int r=0;r<128;r++) {
    float* row=m.data()+((size_t)h*128+r)*128;
    for(int t=0;t<accepted;t++) {
      const float* pt=p.data()+(size_t)t*PEND_GDN; float k[128];
      for(int c=0;c<128;c++) k[c]=pt[(h%HK)*128+c];
      const float g=pt[KDIM+VDIM+h], beta=pt[KDIM+VDIM+HV+h];
      const float v=pt[KDIM+h*128+r]; float innovation=v;
      if(!delta) { float dot=0; for(int c=0;c<128;c++) dot=fmaf(row[c],k[c],dot); innovation=beta*(v-g*dot); }
      for(int c=0;c<128;c++) row[c]=fmaf(g,row[c],innovation*k[c]);
    }
  }
  return m;
}
static void contracts(cudaStream_t st) {
  Inputs in; State q(StateFormat::Q8,TEST_SLOTS);
  for(bool delta:{false,true}) for(int accepted:{0,3,8}) for(int gl:{1,NGDN-1}) {
    CU(cudaMemset(q.committed.p,0,q.committed.bytes)); CU(cudaMemset(q.pending.p,0,q.pending.bytes)); seed_layer(q.committed,StateFormat::Q8,ACTIVE,gl);
    std::vector<std::vector<unsigned char>> before; std::vector<std::vector<float>> initial;
    for(int s=0;s<ACTIVE;s++) { before.push_back(q.committed.get<unsigned char>(sizeof(GdnQ8Layer),(s*(size_t)NGDN+gl)*sizeof(GdnQ8Layer))); initial.push_back(decode_layer(q.committed,StateFormat::Q8,s,gl)); }
    in.segments(8,0,1,0); launch_gdn(q.args(in,gl,delta),ACTIVE,StateFormat::Q8,st); CU(cudaGetLastError()); CU(cudaStreamSynchronize(st));
    for(int s=0;s<ACTIVE;s++) require(before[s]==q.committed.get<unsigned char>(sizeof(GdnQ8Layer),(s*(size_t)NGDN+gl)*sizeof(GdnQ8Layer)),"keep with no replay preserves exact committed bytes");
    std::vector<std::vector<float>> expected;
    for(int s=0;s<ACTIVE;s++) {
      const size_t poff=(s*(size_t)NGDN+gl)*2*PMAX*PEND_GDN*sizeof(__half);
      const auto half=q.pending.get<__half>((size_t)PMAX*PEND_GDN,poff); std::vector<float> p(half.size());
      for(size_t i=0;i<p.size();i++) p[i]=__half2float(half[i]);
      expected.push_back(replay_reference(initial[s],p,accepted,delta));
    }
    in.segments(8,accepted,1,1); launch_gdn(q.args(in,gl,delta),ACTIVE,StateFormat::Q8,st); CU(cudaGetLastError()); CU(cudaStreamSynchronize(st));
    double worst=0;
    for(int s=0;s<ACTIVE;s++) {
      const auto actual=decode_layer(q.committed,StateFormat::Q8,s,gl); const auto ref=encode(expected[s]);
      for(int h=0;h<HV;h++) for(int r=0;r<128;r++) for(int c=0;c<128;c++) {
        const size_t at=((size_t)h*128+r)*128+c; const float value=ref.data[h][r][c]*ref.scales[h][r];
        const double error=fabs(actual[at]-value), bound=2*ref.scales[h][r]+5e-6;
        worst=std::max(worst,error/bound); require(error<=bound,"same-format replay agrees with pending-row reference (two codes + 5e-6)");
      }
    }
    // No state-save boundary is allowed merely because the token staging chunk ends at 16.
    const auto cv=in.conv.get<float>((size_t)ACTIVE*ROWS*CONV_CH),gv=in.gates.get<float>((size_t)ACTIVE*ROWS*2*HV);
    for(int s=0;s<ACTIVE;s++) {
      const int segment=(s+1)%ACTIVE; std::vector<float> p((size_t)ROWS*PEND_GDN);
      for(int t=0;t<ROWS;t++) {
        const size_t row=(size_t)segment*ROWS+t,base=(size_t)t*PEND_GDN;
        std::copy_n(cv.data()+row*CONV_CH+KDIM,KDIM,p.data()+base);
        std::copy_n(cv.data()+row*CONV_CH+2*KDIM,VDIM,p.data()+base+KDIM);
        std::copy_n(gv.data()+row*2*HV,2*HV,p.data()+base+KDIM+VDIM);
      }
      expected[s]=replay_reference(decode_layer(q.committed,StateFormat::Q8,s,gl),p,ROWS,false);
    }
    in.segments(ROWS,0,0,0); launch_gdn(q.args(in,gl,delta),ACTIVE,StateFormat::Q8,st); CU(cudaGetLastError()); CU(cudaStreamSynchronize(st));
    for(int s=0;s<ACTIVE;s++) {
      const auto actual=decode_layer(q.committed,StateFormat::Q8,s,gl); const auto ref=encode(expected[s]);
      for(int h=0;h<HV;h++) for(int r=0;r<128;r++) for(int c=0;c<128;c++) {
        const size_t at=((size_t)h*128+r)*128+c; const float value=ref.data[h][r][c]*ref.scales[h][r];
        require(fabs(actual[at]-value)<=2*ref.scales[h][r]+5e-6,"21-token recurrence agrees with one-save same-format reference");
      }
    }
    const auto bytes=q.committed.get<unsigned char>(q.committed.bytes);
    for(int s=0;s<TEST_SLOTS;s++) for(int l=0;l<NGDN;l++) if(s==ACTIVE||l!=gl) {
      const size_t off=(s*(size_t)NGDN+l)*sizeof(GdnQ8Layer);
      for(size_t i=0;i<sizeof(GdnQ8Layer);i++) require(bytes[off+i]==0,"untouched slots and layers");
    }
    const size_t sb=gdn_state_slot_bytes(StateFormat::Q8);
    CU(cudaMemcpy((char*)q.committed.p+ACTIVE*sb,q.committed.p,sb,cudaMemcpyDeviceToDevice));
    require(q.committed.get<unsigned char>(sb)==q.committed.get<unsigned char>(sb,ACTIVE*sb),"whole-slot copy includes scales");
    CU(cudaMemset((char*)q.committed.p+ACTIVE*sb,0,sb));
    const auto reset=q.committed.get<unsigned char>(sb,ACTIVE*sb); for(auto x:reset) require(x==0,"whole-slot reset includes scales");
    json("{\"kind\":\"contract\",\"layer\":%d,\"accept\":%d,\"replay_delta\":%s,\"worst_bound_fraction\":%.9g,\"status\":\"pass\"}\n",gl,accepted,delta?"true":"false",worst);
  }
}
static void conv_contract(cudaStream_t st) {
  Inputs in;
  Buffer proj((size_t)ACTIVE*ROWS*GDN_IN*4),w((size_t)CONV_CH*4*4),ab((size_t)ACTIVE*ROWS*2*HV*4),ssma(HV*4),dt(HV*4);
  std::vector<float> x((size_t)ACTIVE*ROWS*GDN_IN), weights((size_t)CONV_CH*4), av((size_t)ACTIVE*ROWS*2*HV), a(HV,-.1f),d(HV,.2f);
  for(size_t i=0;i<x.size();i++) x[i]=.4f*sinf((i%8191)*.17f);
  for(size_t i=0;i<weights.size();i++) weights[i]=.1f*(i%4+1);
  for(size_t i=0;i<av.size();i++) av[i]=.3f*sinf(i*.23f);
  proj.put(x.data(),x.size()*4); w.put(weights.data(),weights.size()*4); ab.put(av.data(),av.size()*4); ssma.put(a.data(),HV*4); dt.put(d.data(),HV*4);
  const size_t cs=(size_t)TEST_SLOTS*NGDN*3*CONV_CH*4, ps=(size_t)TEST_SLOTS*NGDN*PMAX*CONV_CH*sizeof(__half), os=(size_t)ACTIVE*ROWS*CONV_CH*4,gs=(size_t)ACTIVE*ROWS*2*HV*4;
  Buffer state0(cs),state1(cs),pend0(ps),pend1(ps),out0(os),out1(os),gate0(gs),gate1(gs);
  ConvArgs ca{}; ca.proj=proj.as<float>(); ca.w=w.as<float>(); ca.ab=ab.as<float>(); ca.ssm_a=ssma.as<float>(); ca.dt_bias=dt.as<float>(); ca.segs=in.segs.as<Seg>(); ca.gl=NGDN-1;
  const int steps[][4]={{8,0,1,0},{8,0,1,1},{8,3,1,0},{21,8,0,1}};
  for(const auto& step:steps) {
    in.segments(step[0],step[1],step[2],step[3]);
    ca.state=state0.as<float>(); ca.pend=pend0.p; ca.out=out0.as<float>(); ca.gates=gate0.as<float>(); launch_conv(ca,ACTIVE,StateFormat::F16,st);
    ca.state=state1.as<float>(); ca.pend=pend1.p; ca.out=out1.as<float>(); ca.gates=gate1.as<float>(); launch_conv(ca,ACTIVE,StateFormat::Q8,st);
    CU(cudaGetLastError()); CU(cudaStreamSynchronize(st));
    const std::pair<const Buffer*,const Buffer*> pairs[]={{&state0,&state1},{&pend0,&pend1},{&out0,&out1},{&gate0,&gate1}};
    for(const auto& pair:pairs)
      require(pair.first->get<unsigned char>(pair.first->bytes)==pair.second->get<unsigned char>(pair.second->bytes),"Q8 conv uses identical half pending as F16");
  }
  json("{\"kind\":\"contract\",\"tensor\":\"conv_state_pending_output_gates\",\"status\":\"pass\"}\n");
}
__global__ void timing_init_half(__half* p,size_t n) {
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x) p[i]=__float2half((int(i%255)-127)*.0002f);
}
__global__ void timing_init_q8(GdnQ8Layer* p,size_t n) {
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n*M;i+=(size_t)blockDim.x*gridDim.x) reinterpret_cast<int8_t*>(p+i/M)[i%M]=int(i%255)-127;
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n*HV*128;i+=(size_t)blockDim.x*gridDim.x) reinterpret_cast<float*>(p[i/(HV*128)].scales)[i%(HV*128)]=.0002f;
}
struct Graph {
  cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
  ~Graph() { if(exec) CU(cudaGraphExecDestroy(exec)); if(graph) CU(cudaGraphDestroy(graph)); }
  Graph()=default; Graph(const Graph&)=delete;
};
static void capture(Graph& graph,GdnArgs a,StateFormat f,int slots,int banks,int layers,bool single,cudaStream_t st) {
  CU(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
  void* base=a.state;
  for(int bank=0;bank<banks;bank++) for(int gl=0;gl<layers;gl++) {
    a.state=(char*)base+(size_t)bank*slots*gdn_state_slot_bytes(f); a.gl=gl;
    if(single) launch_gdn_single(a,slots,f,st); else launch_gdn(a,slots,f,st);
  }
  CU(cudaStreamEndCapture(st,&graph.graph)); CU(cudaGraphInstantiate(&graph.exec,graph.graph,0));
}
static double timed(Graph& graph,int reps,int banks,cudaStream_t st) {
  cudaEvent_t begin,end; CU(cudaEventCreate(&begin)); CU(cudaEventCreate(&end)); CU(cudaEventRecord(begin,st));
  for(int i=0;i<reps;i++) CU(cudaGraphLaunch(graph.exec,st));
  CU(cudaEventRecord(end,st)); CU(cudaEventSynchronize(end)); float ms=0; CU(cudaEventElapsedTime(&ms,begin,end));
  CU(cudaEventDestroy(begin)); CU(cudaEventDestroy(end)); return ms*1000./(reps*banks);
}
static void restore_checkpoint(const Buffer& checkpoint,Buffer& working,cudaStream_t st) {
  require(checkpoint.bytes==working.bytes,"checkpoint and shared working-state extents match");
  CU(cudaMemcpyAsync(working.p,checkpoint.p,working.bytes,cudaMemcpyDeviceToDevice,st));
  CU(cudaStreamSynchronize(st));
}
static double checkpoint_timed(Graph& graph,const Buffer& checkpoint,Buffer& working,int reps,int banks,cudaStream_t st) {
  restore_checkpoint(checkpoint,working,st);
  return timed(graph,reps,banks,st);
}
enum class TimingComparison { Format, Single, ScaleGeneric, ScaleSingle };
static void timing(int slots,int banks,int layers,int reps,int rounds,TimingComparison comparison,const cudaDeviceProp& prop,cudaStream_t st) {
  const bool single=comparison==TimingComparison::Single||comparison==TimingComparison::ScaleSingle;
  const bool scale_abba=comparison==TimingComparison::ScaleGeneric||comparison==TimingComparison::ScaleSingle;
  const StateFormat control_format=single||scale_abba?StateFormat::Q8:StateFormat::F16;
  const size_t controlbytes=(size_t)slots*banks*gdn_state_slot_bytes(control_format),q8bytes=(size_t)slots*banks*gdn_state_slot_bytes(StateFormat::Q8);
  const size_t touched_q8=(size_t)slots*banks*layers*layer_bytes(StateFormat::Q8);
  require(touched_q8>2*(size_t)prop.l2CacheSize,"timing Q8 working set must exceed twice device L2 (increase banks/layers)");
  const size_t pendingbytes=(size_t)slots*NGDN*2*PMAX*PEND_GDN*sizeof(__half);
  size_t freebytes,total; CU(cudaMemGetInfo(&freebytes,&total)); require(controlbytes+q8bytes+pendingbytes+64*1024*1024<freebytes,"timing memory budget fits available GPU RAM");
  Buffer control(controlbytes),q8(q8bytes),co((size_t)slots*CONV_CH*4),gate((size_t)slots*2*HV*4),y((size_t)slots*VDIM*4),sg(slots*sizeof(Seg));
  const auto input=Inputs(); const auto co1=input.conv.get<float>(CONV_CH),g1=input.gates.get<float>(2*HV);
  std::vector<float> cv((size_t)slots*CONV_CH),gv((size_t)slots*2*HV); std::vector<Seg> seg(slots);
  for(int s=0;s<slots;s++) { std::copy(co1.begin(),co1.end(),cv.begin()+(size_t)s*CONV_CH); std::copy(g1.begin(),g1.end(),gv.begin()+(size_t)s*2*HV); seg[s]={s,s,1,0,0,0}; }
  co.put(cv.data(),cv.size()*4); gate.put(gv.data(),gv.size()*4); sg.put(seg.data(),seg.size()*sizeof(Seg));
  timing_init_q8<<<256,256,0,st>>>(q8.as<GdnQ8Layer>(),(size_t)slots*banks*NGDN);
  if(single||scale_abba) CU(cudaMemcpyAsync(control.p,q8.p,q8bytes,cudaMemcpyDeviceToDevice,st));
  else timing_init_half<<<256,256,0,st>>>(control.as<__half>(),controlbytes/sizeof(__half));
  CU(cudaGetLastError()); CU(cudaStreamSynchronize(st));
  // Even len=1/no replay computes pending pointers. Allocate the full typed extent rather than relying on null pointer arithmetic.
  Buffer pending(pendingbytes);
  GdnArgs a{}; a.conv_out=co.as<float>(); a.gates=gate.as<float>(); a.y=y.as<float>(); a.segs=sg.as<Seg>(); a.pend=pending.p; a.q8_scale=gdn_q8_scale_mode();
  Graph fgraph,qgraph;
  if(scale_abba) {
    a.state=q8.p; a.q8_scale=Q8ScaleMode::Max;
    capture(fgraph,a,StateFormat::Q8,slots,banks,layers,single,st);
    a.q8_scale=Q8ScaleMode::LeastSquares;
    capture(qgraph,a,StateFormat::Q8,slots,banks,layers,single,st);
  } else {
    a.state=control.p; capture(fgraph,a,control_format,slots,banks,layers,false,st);
    a.state=q8.p; capture(qgraph,a,StateFormat::Q8,slots,banks,layers,single,st);
  }
  for(int i=0;i<3;i++) for(Graph* graph:{&fgraph,&qgraph}) {
    if(scale_abba) restore_checkpoint(control,q8,st);
    CU(cudaGraphLaunch(graph->exec,st));
  }
  CU(cudaStreamSynchronize(st));
  double fsum=0,qsum=0;
  for(int r=0;r<rounds;r++) {
    const auto member=[&](Graph& graph) { return scale_abba?checkpoint_timed(graph,control,q8,reps,banks,st):timed(graph,reps,banks,st); };
    const double fa=member(fgraph),qa=member(qgraph),qb=member(qgraph),fb=member(fgraph);
    fsum+=(fa+fb)/2; qsum+=(qa+qb)/2;
    if(scale_abba) paired_json("{\"kind\":\"abba\",\"comparison\":\"max_vs_ls\",\"control_mode\":\"max\",\"candidate_mode\":\"ls\",\"route\":\"%s\",\"arithmetic\":\"fp32\",\"shared_state_addresses\":true,\"checkpoint_restore_bytes\":%zu,\"restore_outside_timing\":true,\"slots\":%d,\"round\":%d,\"max_a_us\":%.9g,\"ls_a_us\":%.9g,\"ls_b_us\":%.9g,\"max_b_us\":%.9g}\n",single?"single_q8":"generic_q8",q8bytes,slots,r,fa,qa,qb,fb);
    else if(single) json("{\"kind\":\"abba\",\"control_route\":\"generic_q8\",\"candidate_route\":\"single_q8\",\"arithmetic\":\"fp32\",\"slots\":%d,\"round\":%d,\"generic_q8_a_us\":%.9g,\"single_q8_a_us\":%.9g,\"single_q8_b_us\":%.9g,\"generic_q8_b_us\":%.9g}\n",slots,r,fa,qa,qb,fb);
    else json("{\"kind\":\"abba\",\"slots\":%d,\"round\":%d,\"f16_a_us\":%.9g,\"q8_a_us\":%.9g,\"q8_b_us\":%.9g,\"f16_b_us\":%.9g}\n",slots,r,fa,qa,qb,fb);
  }
  const double ft=fsum/rounds,qt=qsum/rounds;
  const size_t fb=(size_t)slots*layers*layer_bytes(control_format),qb=(size_t)slots*layers*layer_bytes(StateFormat::Q8);
  const double ceiling=1504.835e9,ffloor=2.*fb/ceiling*1e6,qfloor=2.*qb/ceiling*1e6;
  if(scale_abba) {
    paired_json("{\"kind\":\"timing\",\"comparison\":\"max_vs_ls\",\"control_mode\":\"max\",\"candidate_mode\":\"ls\",\"route\":\"%s\",\"arithmetic\":\"fp32\",\"shared_state_addresses\":true,\"restore_outside_timing\":true,\"conditioning\":\"identical_full_checkpoint_copy\",\"slots\":%d,\"banks\":%d,\"layers\":%d,\"reps\":%d,\"rounds\":%d,\"q8_working_set_bytes\":%zu,\"l2_bytes\":%d,\"shared_state_allocation_bytes\":%zu,\"checkpoint_allocation_bytes\":%zu,\"checkpoint_restore_bytes_per_member\":%zu,\"step_state_bytes_including_scales\":%zu,\"step_scale_bytes\":%zu,\"max_us\":%.9g,\"ls_us\":%.9g,\"ls_over_max\":%.9g,\"ls_cost_percent\":%.9g,\"max_effective_state_GBs\":%.9g,\"ls_effective_state_GBs\":%.9g,\"assumed_read_write_ceiling_GBs\":1504.835,\"q8_state_floor_us\":%.9g,\"max_gap_us\":%.9g,\"ls_gap_us\":%.9g}\n",single?"single_q8":"generic_q8",slots,banks,layers,reps,rounds,touched_q8,prop.l2CacheSize,q8bytes,controlbytes,q8bytes,qb,(size_t)slots*layers*HV*128*sizeof(float),ft,qt,qt/ft,100.*(qt/ft-1.),2.*qb/(ft*1000),2.*qb/(qt*1000),qfloor,ft-qfloor,qt-qfloor);
    return;
  }
  if(single) {
    json("{\"kind\":\"timing\",\"control_route\":\"generic_q8\",\"candidate_route\":\"single_q8\",\"arithmetic\":\"fp32\",\"slots\":%d,\"banks\":%d,\"layers\":%d,\"reps\":%d,\"rounds\":%d,\"q8_working_set_bytes_per_route\":%zu,\"l2_bytes\":%d,\"generic_q8_alloc_bytes\":%zu,\"single_q8_alloc_bytes\":%zu,\"q8_step_state_bytes_including_scales_per_route\":%zu,\"q8_step_scale_bytes_per_route\":%zu,\"generic_q8_us\":%.9g,\"single_q8_us\":%.9g,\"speedup\":%.9g,\"generic_q8_effective_state_GBs\":%.9g,\"single_q8_effective_state_GBs\":%.9g,\"assumed_read_write_ceiling_GBs\":1504.835,\"q8_state_floor_us_per_route\":%.9g,\"generic_q8_gap_us\":%.9g,\"single_q8_gap_us\":%.9g}\n",slots,banks,layers,reps,rounds,touched_q8,prop.l2CacheSize,controlbytes,q8bytes,qb,(size_t)slots*layers*HV*128*sizeof(float),ft,qt,ft/qt,2.*fb/(ft*1000),2.*qb/(qt*1000),qfloor,ft-ffloor,qt-qfloor);
    return;
  }
  json("{\"kind\":\"timing\",\"slots\":%d,\"banks\":%d,\"layers\":%d,\"reps\":%d,\"rounds\":%d,\"q8_working_set_bytes\":%zu,\"l2_bytes\":%d,\"f16_alloc_bytes\":%zu,\"q8_alloc_bytes\":%zu,\"f16_step_state_bytes\":%zu,\"q8_step_state_bytes_including_scales\":%zu,\"q8_step_scale_bytes\":%zu,\"f16_us\":%.9g,\"q8_us\":%.9g,\"speedup\":%.9g,\"f16_effective_state_GBs\":%.9g,\"q8_effective_state_GBs\":%.9g,\"assumed_read_write_ceiling_GBs\":1504.835,\"f16_state_floor_us\":%.9g,\"q8_state_floor_us\":%.9g,\"f16_gap_us\":%.9g,\"q8_gap_us\":%.9g}\n",slots,banks,layers,reps,rounds,touched_q8,prop.l2CacheSize,controlbytes,q8bytes,fb,qb,(size_t)slots*layers*HV*128*sizeof(float),ft,qt,ft/qt,2.*fb/(ft*1000),2.*qb/(qt*1000),ffloor,qfloor,ft-ffloor,qt-qfloor);
}
static int number(const char* s,int low,int high) { char* end=nullptr; long n=strtol(s,&end,10); require(end!=s&&!*end&&n>=low&&n<=high,"integer option range"); return int(n); }
int main(int argc,char** argv) {
  enum class Run { All, Contracts, Timing, Scale, Selector, ScaleAbba };
  Run run=Run::All; bool single=false; int reps=10,rounds=3,banks=8,layers=NGDN;
  for(int i=1;i<argc;i++) {
    if(!strcmp(argv[i],"--help")) { puts("gdn-q8 [--single] [--contracts-only|--timing-only|--scale-only|--selector-only|--scale-abba] [--reps 1..100] [--rounds 1..10] [--banks 1..64] [--layers 1..48]"); return 0; }
    if(!strcmp(argv[i],"--single")) single=true;
    else if(!strcmp(argv[i],"--contracts-only")||!strcmp(argv[i],"--timing-only")||!strcmp(argv[i],"--scale-only")||!strcmp(argv[i],"--selector-only")||!strcmp(argv[i],"--scale-abba")) {
      require(run==Run::All,"exclusive run selection");
      if(!strcmp(argv[i],"--contracts-only")) run=Run::Contracts;
      else if(!strcmp(argv[i],"--timing-only")) run=Run::Timing;
      else if(!strcmp(argv[i],"--scale-only")) run=Run::Scale;
      else if(!strcmp(argv[i],"--scale-abba")) run=Run::ScaleAbba;
      else run=Run::Selector;
    }
    else if(i+1<argc&&!strcmp(argv[i],"--reps")) reps=number(argv[++i],1,100);
    else if(i+1<argc&&!strcmp(argv[i],"--rounds")) rounds=number(argv[++i],1,10);
    else if(i+1<argc&&!strcmp(argv[i],"--banks")) banks=number(argv[++i],1,64);
    else if(i+1<argc&&!strcmp(argv[i],"--layers")) layers=number(argv[++i],1,NGDN);
    else { fprintf(stderr,"unknown or incomplete option: %s\n",argv[i]); return 3; }
  }
  require(!single||(run!=Run::Scale&&run!=Run::Selector),"scale/selector suites do not take --single");
  cudaStream_t st=nullptr;
  try {
    if(run==Run::Selector) { selector_contracts(); return 0; }
    (void)gdn_q8_scale_mode();
    if(setenv("H5_GDN_ARITH","fp32",1)) { perror("setenv"); return 2; }
    cudaDeviceProp prop{}; CU(cudaGetDeviceProperties(&prop,0));
    const auto device_json=run==Run::ScaleAbba?paired_json:json;
    device_json("{\"kind\":\"device\",\"name\":\"%s\",\"q8_layer_data_bytes\":%zu,\"q8_layer_scale_bytes\":%zu,\"q8_slot_bytes\":%zu,\"f16_slot_bytes\":%zu}\n",prop.name,M,M/128*sizeof(float),gdn_state_slot_bytes(StateFormat::Q8),gdn_state_slot_bytes(StateFormat::F16));
    CU(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    if(run==Run::All||run==Run::Contracts||run==Run::Scale) scale_contracts(st);
    if(run==Run::All||run==Run::Contracts) { if(single) single_contracts(st); else { contracts(st); conv_contract(st); quality(st); } }
    if(run==Run::All||run==Run::Timing||run==Run::ScaleAbba) {
      const TimingComparison comparison=run==Run::ScaleAbba?(single?TimingComparison::ScaleSingle:TimingComparison::ScaleGeneric):(single?TimingComparison::Single:TimingComparison::Format);
      timing(1,banks,layers,reps,rounds,comparison,prop,st); timing(64,1,layers,reps,rounds,comparison,prop,st);
    }
  } catch(const std::exception& e) { fprintf(stderr,"launch error: %s\n",e.what()); if(st) CU(cudaStreamDestroy(st)); return 2; }
  CU(cudaStreamDestroy(st)); return 0;
}
