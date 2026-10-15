#include "gdn_verify_metrics.cuh"
namespace h5 {
void launch_conv(const ConvArgs&, int, StateFormat, cudaStream_t);
void launch_conv_gdn(const ConvArgs&, const GdnArgs&, int, StateFormat, cudaStream_t);
}
constexpr size_t CS = (size_t)3 * CONV_CH, CP = (size_t)CAP * CONV_CH;
enum class Reference { GenericSeparate, SerialSeparate, GenericFused };
static void launch_reference(const ConvArgs& c,const GdnArgs& a,int rows,Reference route,cudaStream_t st) {
  if (route==Reference::GenericFused) launch_conv_gdn(c,a,1,StateFormat::F16,st);
  else {
    launch_conv(c,1,StateFormat::F16,st);
    launch_gdn_verify(a,1,rows,route==Reference::SerialSeparate?GdnVerifyMode::Serial:GdnVerifyMode::Off,StateFormat::F16,st);
  }
}
struct FusedGraph {
  cudaGraph_t graph; cudaGraphExec_t exec;
  FusedGraph(const ConvArgs& c,const GdnArgs& a,int rows,Reference route,bool candidate,cudaStream_t st) {
    CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
    for (int i=0;i<32;++i) {
      if (candidate) launch_conv_gdn_verify(c,a,1,rows,StateFormat::F16,st);
      else launch_reference(c,a,rows,route,st);
    }
    CHECK(cudaStreamEndCapture(st,&graph)); CHECK(cudaGraphInstantiate(&exec,graph,0));
  }
  ~FusedGraph() { CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph)); }
};
static double elapsed(const FusedGraph& g,int launches,cudaStream_t st,cudaEvent_t begin,cudaEvent_t end) {
  CHECK(cudaEventRecord(begin,st));
  for (int i=0;i<launches;++i) CHECK(cudaGraphLaunch(g.exec,st));
  CHECK(cudaEventRecord(end,st)); CHECK(cudaEventSynchronize(end));
  float ms; CHECK(cudaEventElapsedTime(&ms,begin,end)); return ms*1000./(32*launches);
}
static double median(std::vector<double> x) { std::sort(x.begin(),x.end()); return x[x.size()/2]; }
static void paired(const ConvArgs& c,const GdnArgs& a,const ConvArgs& cf,const GdnArgs& af,
    int rows,int accept,Reference route,const char* name,int launches,int pairs,cudaStream_t st) {
  FusedGraph ref(c,a,rows,route,false,st),test(cf,af,rows,route,true,st);
  for (int i=0;i<4;++i) { CHECK(cudaGraphLaunch(ref.exec,st)); CHECK(cudaGraphLaunch(test.exec,st)); }
  cudaEvent_t begin,end; CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
  std::vector<double> rs,ts,ds;
  for (int p=0;p<pairs;++p) {
    double r=0,t=0;
    for (int k=0;k<4;++k) {
      const bool candidate=((k==1||k==2)!=bool(p&1));
      (candidate?t:r)+=.5*elapsed(candidate?test:ref,launches,st,begin,end);
    }
    rs.push_back(r); ts.push_back(t); ds.push_back((t/r-1)*100.);
    printf("{\"kind\":\"pair\",\"mode\":\"fused\",\"reference\":\"%s\",\"rows\":%d,\"accept\":%d,\"replay_delta\":%s,\"pair\":%d,\"order\":\"%s\",\"ref_us\":%.9g,\"candidate_us\":%.9g,\"delta_pct\":%.9g}\n",
      name,rows,accept,a.replay_delta?"true":"false",p,p&1?"BAAB":"ABBA",r,t,ds.back());
  }
  printf("{\"kind\":\"timing\",\"scope\":\"warm_layer_repeated_visit\",\"mode\":\"fused\",\"reference\":\"%s\",\"rows\":%d,\"accept\":%d,\"replay_delta\":%s,\"pairs\":%d,\"ref_us\":%.9g,\"candidate_us\":%.9g,\"paired_delta_pct\":%.9g}\n",
    name,rows,accept,a.replay_delta?"true":"false",pairs,median(rs),median(ts),median(ds));
  CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end));
}
static int number(const char* s,int low,int high) {
  char* end=nullptr; long v=strtol(s,&end,10);
  if (end==s||*end||v<low||v>high) throw std::runtime_error("invalid integer argument"); return (int)v;
}
int main(int argc,char** argv) {
  if (argc<2||argc>5 || (strcmp(argv[1],"generic-separate")&&strcmp(argv[1],"serial-separate")&&strcmp(argv[1],"generic-fused"))) {
    fprintf(stderr,"usage: gdn-verify-fused-bench generic-separate|serial-separate|generic-fused [launches=4] [pairs=9] [replay_delta=0|1]\n"); return 3;
  }
  try {
    const int launches=argc>2?number(argv[2],1,100):4,pairs=argc>3?number(argv[3],1,101):9;
    const bool delta=argc>4?number(argv[4],0,1):false;
    const Reference route=!strcmp(argv[1],"generic-fused")?Reference::GenericFused:
      !strcmp(argv[1],"serial-separate")?Reference::SerialSeparate:Reference::GenericSeparate;
    const bool banked=route==Reference::GenericFused;
    if (setenv("H5_PDL","0",1)) throw std::runtime_error("cannot select serialized benchmark launches");
    CHECK(cudaSetDevice(0)); cudaStream_t st; CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,0));
    printf("{\"kind\":\"configuration\",\"device\":\"%s\",\"mode\":\"fused\",\"reference\":\"%s\",\"cg\":\"%s\",\"pdl\":false,\"pending_capacity\":16,\"candidate_conv_banks\":2}\n",
      prop.name,argv[1],getenv("H5_GDN_CG")?getenv("H5_GDN_CG"):"4");
    auto proj=alloc<float>((size_t)ROWS*GDN_IN),w=alloc<float>((size_t)CONV_CH*4),ab=alloc<float>((size_t)ROWS*96);
    auto sa=alloc<float>(HV),dt=alloc<float>(HV),co=alloc<float>((size_t)ROWS*CONV_CH),gates=alloc<float>((size_t)ROWS*96);
    auto m0=alloc<__half>(MS),m1=alloc<__half>(MS),p0=alloc<__half>(PS),p1=alloc<__half>(PS);
    auto c0=alloc<float>((banked?2:1)*CS),c1=alloc<float>(2*CS);
    auto cp0=alloc<__half>((banked?2:1)*CP),cp1=alloc<__half>(2*CP);
    auto y0=alloc<float>((size_t)ROWS*VDIM),y1=alloc<float>((size_t)ROWS*VDIM),read_cs=alloc<float>(CS);
    auto read_cp=alloc<__half>(CP); auto seg=alloc<Seg>(1);
    std::vector<float> pd((size_t)ROWS*GDN_IN),wd((size_t)CONV_CH*4),abd((size_t)ROWS*96),sad(HV),dtd(HV);
    std::vector<float> ci((banked?2:1)*CS),cf(2*CS); std::vector<__half> mi(MS),zero(PS,__float2half(0.f)),cz0((banked?2:1)*CP,__float2half(0.f)),cz1(2*CP,__float2half(0.f));
    for (size_t i=0;i<pd.size();++i) pd[i]=.15f*sinf(float(i%65521)*.23f);
    for (size_t i=0;i<wd.size();++i) wd[i]=.1f+.02f*(i%4);
    for (size_t i=0;i<abd.size();++i) abd[i]=.3f*cosf(float(i)*.13f);
    for (int h=0;h<HV;++h) { sad[h]=-.5f-.01f*h; dtd[h]=.02f*(h%7); }
    for (size_t i=0;i<2*CS;++i) cf[i]=.1f*sinf(float(i%CS)*.17f);
    std::copy(cf.begin(),cf.begin()+ci.size(),ci.begin());
    for (size_t i=0;i<MS;++i) mi[i]=__float2half(.03f*sinf(float(i%8191)*.17f));
    upload(proj,pd,st); upload(w,wd,st); upload(ab,abd,st); upload(sa,sad,st); upload(dt,dtd,st);
    ConvArgs c{}; c.proj=proj; c.w=w; c.state=c0; c.out=co; c.segs=seg; c.pend=cp0;
    c.ab=ab; c.ssm_a=sa; c.dt_bias=dt; c.gates=gates; c.pending_rows=CAP;
    ConvArgs cc=c; cc.state=c1; cc.pend=cp1;
    GdnArgs a{}; a.conv_out=co; a.gates=gates; a.state=m0; a.pend=p0; a.y=y0; a.segs=seg; a.pending_rows=CAP; a.replay_delta=delta;
    GdnArgs b=a; b.state=m1; b.pend=p1; b.y=y1; b.conv_out=nullptr; b.gates=nullptr;
    bool ok=true;
    auto reset=[&]() { upload(c0,ci,st); upload(c1,cf,st); upload(cp0,cz0,st); upload(cp1,cz1,st); upload(m0,mi,st); upload(m1,mi,st); upload(p0,zero,st); upload(p1,zero,st); };
    auto visit=[&](int rows,int accept,int parity,int bank) {
      const Seg sg{0,0,rows,accept,1,parity,bank};
      CHECK(cudaMemcpyAsync(seg,&sg,sizeof(sg),cudaMemcpyHostToDevice,st));
      CHECK(cudaMemcpyAsync(read_cs,c1+bank*CS,CS*4,cudaMemcpyDeviceToDevice,st));
      CHECK(cudaMemcpyAsync(read_cp,cp1+(parity^1)*CP,CP*sizeof(__half),cudaMemcpyDeviceToDevice,st));
      CHECK(cudaStreamSynchronize(st));
      launch_reference(c,a,rows,route,st); launch_conv_gdn_verify(cc,b,1,rows,StateFormat::F16,st); CHECK(cudaGetLastError());
      // The generic fused executable has its own small rounding differences from the separate
      // oracle. Keep it as a numerical/performance control, not the bit-exact acceptance oracle.
      ok=report("fused",rows,accept,delta,"output",y0,y1,(size_t)rows*VDIM,st,!banked)&&ok;
      ok=report("fused",rows,accept,delta,"committed_state",m0,m1,MS,st,!banked)&&ok;
      ok=report("fused",rows,accept,delta,"pending_both_banks",p0,p1,PS,st,!banked)&&ok;
      ok=report("fused",rows,accept,delta,"conv_committed",c0+(banked?(bank^1)*CS:0),c1+(bank^1)*CS,CS,st)&&ok;
      ok=report("fused",rows,accept,delta,"conv_pending",cp0+(banked?parity*CP:0),cp1+parity*CP,(size_t)rows*CONV_CH,st)&&ok;
      ok=report("fused",rows,accept,delta,"conv_read_bank_immutable",read_cs,c1+bank*CS,CS,st)&&ok;
      ok=report("fused",rows,accept,delta,"conv_replay_bank_immutable",read_cp,cp1+(parity^1)*CP,CP,st)&&ok;
    };
    for (int rows : {8,16}) for (int accept : {0,3,7,15}) {
      reset(); visit(16,0,0,0); visit(rows,accept,1,1); visit(rows,std::min(rows,accept),0,0);
      reset(); visit(16,0,0,0);
      const Seg timing{0,0,rows,accept,1,1,1}; CHECK(cudaMemcpyAsync(seg,&timing,sizeof(timing),cudaMemcpyHostToDevice,st)); CHECK(cudaStreamSynchronize(st));
      paired(c,a,cc,b,rows,accept,route,argv[1],launches,pairs,st);
    }
    printf("{\"kind\":\"result\",\"bit_exact_required\":%s,\"conv_bank_exact_required\":true,\"passed\":%s}\n",banked?"false":"true",ok?"true":"false");
    CHECK(cudaFree(proj)); CHECK(cudaFree(w)); CHECK(cudaFree(ab)); CHECK(cudaFree(sa)); CHECK(cudaFree(dt)); CHECK(cudaFree(co)); CHECK(cudaFree(gates));
    CHECK(cudaFree(m0)); CHECK(cudaFree(m1)); CHECK(cudaFree(p0)); CHECK(cudaFree(p1)); CHECK(cudaFree(c0)); CHECK(cudaFree(c1)); CHECK(cudaFree(cp0)); CHECK(cudaFree(cp1)); CHECK(cudaFree(y0)); CHECK(cudaFree(y1)); CHECK(cudaFree(read_cs)); CHECK(cudaFree(read_cp)); CHECK(cudaFree(seg));
    CHECK(cudaStreamDestroy(st)); return ok?0:1;
  } catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 2; }
}
