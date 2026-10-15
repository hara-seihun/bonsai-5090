#include "gdn_verify_metrics.cuh"
struct Graph {
  cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
  Graph(const GdnArgs& a,int rows,GdnVerifyMode mode,cudaStream_t st,int nodes) {
    CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
    for (int i=0;i<nodes;++i) launch_gdn_verify(a,1,rows,mode,StateFormat::F16,st);
    CHECK(cudaStreamEndCapture(st,&graph)); CHECK(cudaGraphInstantiate(&exec,graph,0));
  }
  ~Graph() { CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph)); }
};
static double time_graph(const Graph& g,cudaStream_t st,cudaEvent_t begin,cudaEvent_t end,int launches,int nodes) {
  CHECK(cudaEventRecord(begin,st));
  for (int i=0;i<launches;++i) CHECK(cudaGraphLaunch(g.exec,st));
  CHECK(cudaEventRecord(end,st)); CHECK(cudaEventSynchronize(end));
  float ms; CHECK(cudaEventElapsedTime(&ms,begin,end)); return ms*1000./(launches*nodes);
}
static double median(std::vector<double> x) { std::sort(x.begin(),x.end()); return x[x.size()/2]; }
static void paired(const GdnArgs& a,const GdnArgs& b,int rows,int accept,const char* name,
    GdnVerifyMode mode,int launches,int pairs,cudaStream_t st) {
  constexpr int nodes=32;
  Graph ref(a,rows,GdnVerifyMode::Off,st,nodes),test(b,rows,mode,st,nodes);
  for (int i=0;i<4;++i) { CHECK(cudaGraphLaunch(ref.exec,st)); CHECK(cudaGraphLaunch(test.exec,st)); }
  cudaEvent_t begin,end; CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
  std::vector<double> rs,ts,ds;
  // ABBA and BAAB quartets share one process, stream, input arrays and CG selection.
  // Replay timing repeats the visit, advancing its committed state. Work/branches are value-independent.
  for (int p=0;p<pairs;++p) {
    double r=0,t=0;
    for (int k=0;k<4;++k) {
      const bool candidate=((k==1 || k==2) != bool(p&1));
      const double us=time_graph(candidate?test:ref,st,begin,end,launches,nodes);
      (candidate?t:r)+=us*.5;
    }
    rs.push_back(r); ts.push_back(t); ds.push_back((t/r-1)*100.);
    printf("{\"kind\":\"pair\",\"mode\":\"%s\",\"rows\":%d,\"accept\":%d,\"replay_delta\":%s,\"pair\":%d,\"order\":\"%s\",\"ref_us\":%.9g,\"candidate_us\":%.9g,\"delta_pct\":%.9g}\n",
      name,rows,accept,a.replay_delta?"true":"false",p,p&1?"BAAB":"ABBA",r,t,ds.back());
  }
  printf("{\"kind\":\"timing\",\"scope\":\"warm_layer_repeated_visit\",\"mode\":\"%s\",\"rows\":%d,\"accept\":%d,\"replay_delta\":%s,\"pairs\":%d,\"nodes_per_graph\":%d,\"launches_per_sample\":%d,\"ref_us\":%.9g,\"candidate_us\":%.9g,\"paired_delta_pct\":%.9g}\n",
    name,rows,accept,a.replay_delta?"true":"false",pairs,nodes,launches,median(rs),median(ts),median(ds));
  CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end));
}
static int number(const char* s,int low,int high) {
  char* end=nullptr; long v=strtol(s,&end,10);
  if (end==s || *end || v<low || v>high) throw std::runtime_error("invalid integer argument");
  return (int)v;
}
int main(int argc,char** argv) {
  if (argc<2 || argc>5 || (strcmp(argv[1],"serial") && strcmp(argv[1],"chunk"))) {
    fprintf(stderr,"usage: gdn-verify-bench serial|chunk [graph launches/sample=4] [pairs=9] [replay_delta=0|1]\n"); return 3;
  }
  try {
    const int launches=argc>2?number(argv[2],1,100):4, pairs=argc>3?number(argv[3],1,101):9;
    const bool delta=argc>4?number(argv[4],0,1):false;
    const GdnVerifyMode mode=!strcmp(argv[1],"serial")?GdnVerifyMode::Serial:GdnVerifyMode::Chunk;
    // Back-to-back visits write the same state, unlike the production conv -> GDN boundary.
    // Disable dependent launch so state preloads cannot race the preceding visit's checkpoint.
    if (setenv("H5_PDL","0",1)) throw std::runtime_error("cannot select serialized benchmark launches");
    CHECK(cudaSetDevice(0)); cudaStream_t st; CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,0));
    printf("{\"kind\":\"configuration\",\"device\":\"%s\",\"mode\":\"%s\",\"cg\":\"%s\",\"pending_capacity\":16,\"pdl\":false,\"state\":\"f16\",\"arith\":\"fp32\"}\n",
      prop.name,argv[1],getenv("H5_GDN_CG")?getenv("H5_GDN_CG"):"4");
    __half* m0=alloc<__half>(MS),*m1=alloc<__half>(MS),*p0=alloc<__half>(PS),*p1=alloc<__half>(PS);
    float* co=alloc<float>((size_t)ROWS*CONV_CH),*gs=alloc<float>((size_t)ROWS*96);
    float* y0=alloc<float>((size_t)ROWS*VDIM),*y1=alloc<float>((size_t)ROWS*VDIM);
    Seg* segs=alloc<Seg>(1);
    std::vector<__half> initial(MS),empty(PS,__float2half(0.f));
    for (size_t i=0;i<MS;++i) initial[i]=__float2half(.03f*sinf(float(i%8191)*.17f));
    std::vector<float> data((size_t)ROWS*CONV_CH),gates((size_t)ROWS*96);
    for (int r=0;r<ROWS;++r) {
      for (int kh=0;kh<HK;++kh) for (int qk=0;qk<2;++qk) {
        const size_t base=(size_t)r*CONV_CH+qk*KDIM+kh*128; float ss=0;
        for (int d=0;d<128;++d) { float v=sinf((r+1)*.19f+d*.31f+kh*.13f+qk*.7f); data[base+d]=v; ss+=v*v; }
        for (int d=0;d<128;++d) data[base+d]/=sqrtf(ss);
      }
      for (int d=0;d<VDIM;++d) data[(size_t)r*CONV_CH+2*KDIM+d]=.5f*sinf(d*.11f+r*.17f);
      for (int h=0;h<HV;++h) { gates[r*96+h]=.93f+.065f*(h%7)/6.f; gates[r*96+HV+h]=.2f+.7f*(h%5)/4.f; }
    }
    upload(co,data,st); upload(gs,gates,st);
    GdnArgs a{}; a.conv_out=co; a.gates=gs; a.state=m0; a.pend=p0; a.y=y0; a.segs=segs;
    a.pending_rows=CAP; a.replay_delta=delta;
    GdnArgs b=a; b.state=m1; b.pend=p1; b.y=y1;
    bool ok=true;
    auto visit=[&](int rows,int accept,int parity) {
      const Seg sg{0,0,rows,accept,1,parity};
      CHECK(cudaMemcpyAsync(segs,&sg,sizeof(sg),cudaMemcpyHostToDevice,st));
      // Retain the stack-owned metadata until its asynchronous copy completes.
      CHECK(cudaStreamSynchronize(st));
      launch_gdn_verify(a,1,rows,GdnVerifyMode::Off,StateFormat::F16,st);
      launch_gdn_verify(b,1,rows,mode,StateFormat::F16,st); CHECK(cudaGetLastError());
    };
    for (int rows : {8,16}) for (int accept : {0,3,7,15}) {
      upload(m0,initial,st); upload(m1,initial,st); upload(p0,empty,st); upload(p1,empty,st);
      visit(16,0,0); // valid previous pending rows for acceptance15, including R8 next visits
      ok=report(argv[1],16,-1,delta,"seed_output",y0,y1,(size_t)16*VDIM,st)&&ok;
      ok=report(argv[1],16,-1,delta,"seed_committed",m0,m1,MS,st)&&ok;
      ok=report(argv[1],16,-1,delta,"seed_pending",p0,p1,PS,st)&&ok;
      visit(rows,accept,1);
      ok=report(argv[1],rows,accept,delta,"output",y0,y1,(size_t)rows*VDIM,st)&&ok;
      ok=report(argv[1],rows,accept,delta,"committed_state",m0,m1,MS,st)&&ok;
      ok=report(argv[1],rows,accept,delta,"pending_both_banks",p0,p1,PS,st)&&ok;
      // Continue with the opposite parity to test that innovations are consumed, not just written.
      visit(rows,std::min(accept,rows),0);
      ok=report(argv[1],rows,accept,delta,"next_output",y0,y1,(size_t)rows*VDIM,st)&&ok;
      ok=report(argv[1],rows,accept,delta,"next_committed",m0,m1,MS,st)&&ok;
      ok=report(argv[1],rows,accept,delta,"next_pending_both_banks",p0,p1,PS,st)&&ok;
      // Re-seed a valid 16-row pending bank, then match both arms' timing inputs.
      upload(m0,initial,st); upload(m1,initial,st); upload(p0,empty,st); upload(p1,empty,st);
      visit(16,0,0);
      CHECK(cudaMemcpyAsync(m1,m0,MS*sizeof(__half),cudaMemcpyDeviceToDevice,st));
      CHECK(cudaMemcpyAsync(p1,p0,PS*sizeof(__half),cudaMemcpyDeviceToDevice,st));
      const Seg timing{0,0,rows,accept,1,1};
      CHECK(cudaMemcpyAsync(segs,&timing,sizeof(timing),cudaMemcpyHostToDevice,st)); CHECK(cudaStreamSynchronize(st));
      paired(a,b,rows,accept,argv[1],mode,launches,pairs,st);
    }
    printf("{\"kind\":\"result\",\"serial_bit_exact_required\":%s,\"passed\":%s}\n",mode==GdnVerifyMode::Serial?"true":"false",ok?"true":"false");
    CHECK(cudaFree(m0)); CHECK(cudaFree(m1)); CHECK(cudaFree(p0)); CHECK(cudaFree(p1));
    CHECK(cudaFree(co)); CHECK(cudaFree(gs)); CHECK(cudaFree(y0)); CHECK(cudaFree(y1)); CHECK(cudaFree(segs));
    CHECK(cudaStreamDestroy(st)); return ok?0:1;
  } catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 2; }
}
