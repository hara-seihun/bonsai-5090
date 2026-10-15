// Fixed-work batch verifier + drafter, with a profiler range around measured rounds.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
extern "C" {
void* h5_create(const char*, int, int, int);
int h5_draft_load(void*, const char*);
void h5_pretune(void*); void h5_capture(void*, int);
void h5_reset(void*, int); void h5_keep(void*, int, int); void h5_accept(void*, int, int);
int h5_step(void*, int, const int*, const int*, const int*, int*);
int h5_draft(void*, int, const int*, const int*, const int*, const int*, const int*, const int*, int*);
void h5_prof_report(void*);
}
static bool integer(const char* s, int lo, int hi, int& out) {
  const char* end = s + std::strlen(s); auto r = std::from_chars(s, end, out);
  return r.ec == std::errc{} && r.ptr == end && out >= lo && out <= hi;
}
static bool cuda_ok(cudaError_t err, const char* op) {
  if (err == cudaSuccess) return true;
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(err)); return false;
}
int main(int argc, char** argv) {
  int S, C, steps, accept, R;
  const bool plain = argc == 9 && !std::strcmp(argv[8],"plain");
  if ((argc != 8 && !plain) || !integer(argv[3],1,96,S) || !integer(argv[4],1,8192,C)
      || !integer(argv[5],1,1000,steps) || !integer(argv[6],0,7,accept)
      || !integer(argv[7],1,8,R) || accept >= R || (plain && (R != 1 || accept != 0))) {
    std::fprintf(stderr,"usage: batch-profile MODEL DRAFTER STREAMS CTX STEPS ACCEPT ROWS [plain]\nplain requires ACCEPT=0 ROWS=1\n"); return 2;
  }
  const char* phase = std::getenv("H5_PROFILE_PHASE");
  if (!phase) phase = "round";
  if ((std::strcmp(phase,"round") && std::strcmp(phase,"verify") && std::strcmp(phase,"draft"))
      || (std::strcmp(phase,"round") && steps != 1) || (plain && !std::strcmp(phase,"draft"))) {
    std::fprintf(stderr,"H5_PROFILE_PHASE must be round, or verify/draft with STEPS=1\n"); return 2;
  }
  void* e = h5_create(argv[1], S, C + (steps + 4) * 8 + 16, S * 8);
  if (!e || (!plain && h5_draft_load(e,argv[2]))) { std::fprintf(stderr,"load failed\n");return 1; }
  h5_capture(e,plain?0:1); h5_pretune(e);
  std::vector<int> sl(S),len(S,C),ni(S,accept+1),ir(S),ip(S),an(S,1000),dst(S),dr(S*7,1000);
  std::vector<int> rs(S*R),rt(S*R),rp(S*R),ro(S*R);
  for (int s=0;s<S;s++) { sl[s]=s; ir[s]=s*R; h5_reset(e,s); h5_keep(e,s,plain?0:1); }
  using clock=std::chrono::steady_clock;
  double verify=0,draft=0;
  for (int it=-3;it<steps;it++) {
    if (it==0) {
      h5_prof_report(e);
      if (std::strcmp(phase,"draft") && (!cuda_ok(cudaDeviceSynchronize(),"before profile") || !cuda_ok(cudaProfilerStart(),"profile start"))) return 1;
    }
    for(int s=0;s<S;s++) for(int k=0;k<R;k++) {
      rs[s*R+k]=s;rt[s*R+k]=k?dr[s*7+k-1]:an[s];rp[s*R+k]=len[s]+k;
    }
    auto a=clock::now();
    if(h5_step(e,S*R,rs.data(),rt.data(),rp.data(),ro.data())) { std::fprintf(stderr,"step failed\n");return 1; }
    auto b=clock::now();
    if(it==0 && !std::strcmp(phase,"verify") && !cuda_ok(cudaProfilerStop(),"verify profile stop")) return 1;
    for(int s=0;s<S;s++) { if(!plain)h5_accept(e,s,accept+1);ip[s]=len[s];len[s]+=accept+1;dst[s]=len[s];an[s]=ro[s*R+accept]; }
    if(it==0 && !std::strcmp(phase,"draft") && (!cuda_ok(cudaDeviceSynchronize(),"before draft profile") || !cuda_ok(cudaProfilerStart(),"draft profile start"))) return 1;
    if(!plain && h5_draft(e,S,sl.data(),ni.data(),ir.data(),ip.data(),an.data(),dst.data(),dr.data())) { std::fprintf(stderr,"draft failed\n");return 1; }
    auto c=clock::now();
    if(it>=0) { verify+=std::chrono::duration<double>(b-a).count();draft+=std::chrono::duration<double>(c-b).count(); }
  }
  if(std::strcmp(phase,"verify") && !cuda_ok(cudaProfilerStop(),"profile stop")) return 1;
  h5_prof_report(e);
  std::printf("{\"streams\":%d,\"rows_per_stream\":%d,\"context\":%d,\"steps\":%d,\"forced_accept\":%d,\"verify_ms\":%.6f,\"draft_ms\":%.6f,\"round_ms\":%.6f}\n",
    S,R,C,steps,accept,verify*1000/steps,draft*1000/steps,(verify+draft)*1000/steps);
  return 0;
}
