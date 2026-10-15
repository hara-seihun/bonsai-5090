#include <cuda_runtime.h>
#include "../src/model.h"
#include "../src/draft_probe.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern "C" {
void* h5_create(const char*, int, int, int);
int h5_draft_load(void*, const char*);
void h5_pretune(void*); void h5_capture(void*, int);
void h5_reset(void*, int); void h5_keep(void*, int, int); void h5_accept(void*, int, int);
int h5_step(void*, int, const int*, const int*, const int*, int*);
int h5_draft(void*, int, const int*, const int*, const int*, const int*, const int*, const int*, int*);
const float* h5_draft_conf(void*);
int h5_logits(void*, int, float*);
}

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b-a).count();
}
enum class Failure { None, Cuda, Create, Api, Foreign, Io };
struct Error {
  Failure kind = Failure::None;
  const char* op = "";
  int code = 0;
  cudaError_t native = cudaSuccess;
  std::string detail;
  explicit operator bool() const { return kind != Failure::None; }
};
static Error cuda_error(cudaError_t c, const char* op) {
  if (c == cudaSuccess) return {};
  return {Failure::Cuda, op, 0, c, cudaGetErrorString(c)};
}
template<class F> static Error api(const char* op, F f) {
  try {
    const int code = f();
    const auto native = cudaGetLastError();
    if (code) return {Failure::Api, op, code, native, cudaGetErrorString(native)};
    return cuda_error(native, op);
  } catch (const std::exception& ex) {
    const auto native = cudaGetLastError();
    return {Failure::Foreign, op, 0, native, ex.what()};
  }
}
static std::string quote(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
    else if (c < 32) { char buf[7]; std::snprintf(buf,sizeof(buf),"\\u%04x",c); o += buf; }
    else o += char(c);
  }
  return o + '"';
}
static void report(const Error& e, const char* arm, int engine, int iteration) {
  std::printf("{\"event\":\"error\",\"arm\":%s,\"engine\":%d,\"iteration\":%d,\"kind\":%d,\"op\":%s,\"api_code\":%d,\"cuda_code\":%d,\"cuda_name\":%s,\"detail\":%s}\n",
    quote(arm).c_str(),engine,iteration,int(e.kind),quote(e.op).c_str(),e.code,int(e.native),
    quote(cudaGetErrorName(e.native)).c_str(),quote(e.detail).c_str());
  std::fflush(stdout);
}
static bool integer(const char* s, int lo, int hi, int& out) {
  auto end = s + std::strlen(s); auto r = std::from_chars(s,end,out);
  return r.ec == std::errc{} && r.ptr == end && out >= lo && out <= hi;
}
struct Config {
  std::string model, drafter, plan, output, graph;
  int context, rounds, accept, repeats, device;
  bool single, diagnostic, fresh;
  std::string fresh_arm;
};
struct Observation {
  std::vector<int> verify, draft;
  std::vector<unsigned> confidence;
};
static unsigned bits(float x) { unsigned b;static_assert(sizeof(b)==sizeof(x));std::memcpy(&b,&x,sizeof(b));return b; }
static float value(unsigned b) { float x;std::memcpy(&x,&b,sizeof(x));return x; }
static unsigned long long hash_ints(const std::vector<int>& values,unsigned long long h=14695981039346656037ULL) {
  for (int value:values) for (int byte=0;byte<4;byte++) { h^=(unsigned(value)>>(byte*8))&255;h*=1099511628211ULL; }
  return h;
}
struct PrefixCheckpoint {
  unsigned long long input_hash;
  std::vector<int> output;
  std::vector<float> logits;
};
struct Engine {
  void* ptr = nullptr;
  int streams, offset, context, accept;
  std::vector<int> slots, ingest, row0, pos0, anchor, start, row_slot, token, pos, out, drafts;
  double verify_ms = 0, draft_ms = 0;
  Observation observation;
  bool diagnostic, prefix_exact=true;
  std::vector<PrefixCheckpoint> prefix_baseline;
  std::vector<float> probe;
  std::array<std::vector<std::vector<unsigned>>,4> draft_baseline;
  std::array<std::vector<unsigned>,4> draft_probe_buffers;
  unsigned long long round_input_hash=0;
  Engine(int s, int o, const Config& c): streams(s),offset(o),context(c.context),accept(c.accept),
    slots(s),ingest(s),row0(s),pos0(s),anchor(s),start(s),row_slot(s*8),token(s*8),pos(s*8),out(s*8),drafts(s*7),diagnostic(c.diagnostic) {
    if (diagnostic) {
      probe.resize(h5::VOCAB);
      for (auto& baseline:draft_baseline) baseline.resize(c.rounds);
      draft_probe_buffers[0].resize((size_t)s*8*h5::D);
      draft_probe_buffers[1].resize((size_t)s*7*256);
      draft_probe_buffers[2].resize((size_t)s*7*16);
      draft_probe_buffers[3].resize((size_t)s*7*16);
    }
    for (int i=0;i<s;i++) { slots[i]=i; row0[i]=i*8; }
  }
  // Work never depends on an observed token: both geometries see the same global-stream inputs.
  static int fixed_token(int stream, int position) { return 1000 + (stream*131 + position*17)%30000; }
  Error load(const Config& c) {
    auto e = api("h5_create",[&] { ptr=h5_create(c.model.c_str(),streams,context+(c.rounds+4)*8+16,streams*8); return ptr ? 0 : -1; });
    if (e) { e.kind=Failure::Create; return e; }
    if ((e=api("h5_draft_load",[&] { return h5_draft_load(ptr,c.drafter.c_str()); }))) return e;
    if ((e=api("h5_capture/pretune",[&] { h5_capture(ptr,1); h5_pretune(ptr); return 0; }))) return e;
    return cuda_error(cudaDeviceSynchronize(),"after load");
  }
  void rows(int base) {
    for (int s=0;s<streams;s++) for (int k=0;k<8;k++) {
      const int r=s*8+k; row_slot[r]=s; token[r]=fixed_token(offset+s,base+k); pos[r]=base+k;
    }
  }
  Error checkpoint(const char* arm,int sequence,int position) {
    auto e=api("prefix h5_logits last row",[&] { return h5_logits(ptr,127,probe.data()); });
    if (e) return e;
    const auto input=hash_ints(pos,hash_ints(token,hash_ints(row_slot)));
    const size_t index=position/8;
    const bool first=index>=prefix_baseline.size();
    if (first) prefix_baseline.push_back({input,out,probe});
    const auto& baseline=prefix_baseline[index];
    size_t token_differences=0,logit_differences=0,nonfinite=0;long first_token=-1,first_logit=-1;
    double max_abs=0,squared=0;
    for (size_t i=0;i<out.size();i++) if (out[i]!=baseline.output[i]) { token_differences++;if (first_token<0) first_token=long(i); }
    unsigned long long logit_hash=14695981039346656037ULL;
    for (size_t i=0;i<probe.size();i++) {
      const auto current=bits(probe[i]);
      for (int byte=0;byte<4;byte++) { logit_hash^=(current>>(byte*8))&255;logit_hash*=1099511628211ULL; }
      if (current!=bits(baseline.logits[i])) { logit_differences++;if (first_logit<0) first_logit=long(i); }
      if (!std::isfinite(probe[i]) || !std::isfinite(baseline.logits[i])) nonfinite++;
      else { const double d=double(probe[i])-baseline.logits[i];max_abs=std::max(max_abs,std::abs(d));squared+=d*d; }
    }
    prefix_exact=prefix_exact && input==baseline.input_hash && token_differences==0 && logit_differences==0;
    std::printf("{\"event\":\"prefix_checkpoint\",\"arm\":%s,\"sequence\":%d,\"engine_offset\":%d,\"position\":%d,\"baseline_created\":%s,\"input_hash\":%s,\"input_equal\":%s,\"token_differences\":%zu,\"first_token\":%ld,\"logit_stream\":%d,\"logit_position\":%d,\"logit_hash\":%s,\"logit_bit_differences\":%zu,\"first_logit\":%ld,\"logit_max_abs\":%.9g,\"logit_rms\":%.9g,\"nonfinite_pairs\":%zu}\n",
      quote(arm).c_str(),sequence,offset,position,first ? "true" : "false",quote(std::to_string(input)).c_str(),input==baseline.input_hash ? "true" : "false",
      token_differences,first_token,offset+streams-1,position+7,quote(std::to_string(logit_hash)).c_str(),logit_differences,first_logit,
      max_abs,std::sqrt(squared/probe.size()),nonfinite);
    return {};
  }
  Error restore(const char* arm,int sequence) {
    auto e=api("h5_reset/keep",[&] {
      for (int s=0;s<streams;s++) { h5_reset(ptr,s); h5_keep(ptr,s,0); }
      return 0;
    });
    if (e) return e;
    // Pre-wait state loads require the reset writes to have completed before a graph starts.
    if ((e=cuda_error(cudaDeviceSynchronize(),"reset fence before prefix"))) return e;
    // reset does not clear either KV cache. Replay every readable prefix position, including
    // ingest-only drafter calls, before generating any proposals or measuring a round.
    for (int p=0;p<context;p+=8) {
      rows(p);
      if ((e=api("prefix h5_step",[&] { return h5_step(ptr,streams*8,row_slot.data(),token.data(),pos.data(),out.data()); }))) return e;
      if (diagnostic && (e=checkpoint(arm,sequence,p))) return e;
      for (int s=0;s<streams;s++) { ingest[s]=8;pos0[s]=p;anchor[s]=-1;start[s]=p+8; }
      if ((e=api("prefix h5_draft ingest",[&] { return h5_draft(ptr,streams,slots.data(),ingest.data(),row0.data(),pos0.data(),anchor.data(),start.data(),drafts.data()); }))) return e;
    }
    if ((e=api("h5_keep",[&] { for (int s=0;s<streams;s++) h5_keep(ptr,s,1); return 0; }))) return e;
    verify_ms=draft_ms=0; observation={};
    return cuda_error(cudaDeviceSynchronize(),"after restore");
  }
  Error draft_checkpoint(int iteration) {
    const std::array<h5::DraftProbeKind,4> kinds{{h5::DraftProbeKind::TrunkResidual,h5::DraftProbeKind::HiddenProjection,h5::DraftProbeKind::TopValues,h5::DraftProbeKind::TopIndices}};
    const std::array<const char*,4> names{{"trunk_residual","hidden_projection","top_values","top_indices"}};
    for (size_t phase=0;phase<kinds.size();phase++) {
      auto& current=draft_probe_buffers[phase];auto& baseline=draft_baseline[phase][iteration];
      const auto result=h5_draft_probe(ptr,kinds[phase],current.data(),current.size()*sizeof(unsigned));
      if (result.error!=h5::DraftProbeError::None)
        return {Failure::Api,"h5_draft_probe",int(result.error),cudaError_t(result.cuda_code),names[phase]};
      if (result.bytes!=current.size()*sizeof(unsigned) || size_t(result.rows)*result.columns!=current.size())
        return {Failure::Api,"h5_draft_probe contract",-1,cudaSuccess,names[phase]};
      const bool first=baseline.empty();if (first) baseline=current;
      size_t unequal=0,nonfinite=0;long first_element=-1;double max_abs=0,squared=0;
      std::vector<bool> changed(result.rows,false);
      unsigned long long hash=14695981039346656037ULL;
      for (size_t i=0;i<current.size();i++) {
        for (int byte=0;byte<4;byte++) { hash^=(current[i]>>(byte*8))&255;hash*=1099511628211ULL; }
        if (current[i]!=baseline[i]) { unequal++;changed[i/result.columns]=true;if (first_element<0) first_element=long(i); }
        if (phase!=3) {
          const float a=value(baseline[i]),b=value(current[i]);
          if (!std::isfinite(a) || !std::isfinite(b)) nonfinite++;
          else { const double d=double(b)-a;max_abs=std::max(max_abs,std::abs(d));squared+=d*d; }
        }
      }
      std::string ranges="[";bool comma=false;
      for (int row=0;row<result.rows;) {
        if (!changed[row]) { row++;continue; }
        const int begin=row;while (row<result.rows && changed[row]) row++;
        if (comma) ranges+=',';
        comma=true;ranges+='['+std::to_string(begin)+','+std::to_string(row)+']';
      }
      ranges+=']';
      double scale=0,fit_residual=0;
      if (phase!=3 && first_element>=0) {
        const size_t row0=size_t(first_element)/result.columns*result.columns;
        double xx=0,xy=0,yy=0;
        for (int col=0;col<result.columns;col++) { const double a=value(baseline[row0+col]),b=value(current[row0+col]);if (std::isfinite(a)&&std::isfinite(b)) { xx+=a*a;xy+=a*b;yy+=b*b; } }
        if (xx>0) { scale=xy/xx;if (yy>0) fit_residual=std::sqrt(std::max(0.,yy-xy*xy/xx)/yy); }
      }
      std::string groups="[";
      if (phase==1 && first_element>=0) {
        const size_t row0=size_t(first_element)/result.columns*result.columns;
        for (int col0=0;col0<result.columns;col0+=64) {
          double xx=0,xy=0,yy=0;int changed_count=0;
          for (int col=col0;col<std::min(col0+64,result.columns);col++) {
            const double a=value(baseline[row0+col]),b=value(current[row0+col]);
            changed_count+=baseline[row0+col]!=current[row0+col];
            if (std::isfinite(a)&&std::isfinite(b)) { xx+=a*a;xy+=a*b;yy+=b*b; }
          }
          const double group_scale=xx>0 ? xy/xx : 0;
          const double residual=xx>0 && yy>0 ? std::sqrt(std::max(0.,yy-xy*xy/xx)/yy) : 0;
          char item[256];std::snprintf(item,sizeof(item),"%s{\"column\":%d,\"bit_differences\":%d,\"scale\":%.9g,\"fit_residual\":%.9g}",col0 ? "," : "",col0,changed_count,group_scale,residual);
          groups+=item;
        }
      }
      groups+=']';
      std::printf("{\"event\":\"draft_checkpoint\",\"engine_offset\":%d,\"iteration\":%d,\"phase\":%s,\"baseline_created\":%s,\"rows\":%d,\"columns\":%d,\"hash\":%s,\"bit_differences\":%zu,\"first_element\":%ld,\"changed_row_ranges\":%s,\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite_pairs\":%zu,\"first_changed_row_scale\":%.9g,\"first_changed_row_fit_residual\":%.9g,\"first_changed_row_groups\":%s}\n",
        offset,iteration,quote(names[phase]).c_str(),first ? "true" : "false",result.rows,result.columns,quote(std::to_string(hash)).c_str(),unequal,first_element,ranges.c_str(),max_abs,std::sqrt(squared/current.size()),nonfinite,scale,fit_residual,groups.c_str());
    }
    return {};
  }
  Error round(int iteration, bool record) {
    const int base=context+iteration*(accept+1);
    rows(base);
    round_input_hash=hash_ints(pos,hash_ints(token,hash_ints(row_slot)));
    const auto a=Clock::now();
    auto e=api("h5_step",[&] { return h5_step(ptr,streams*8,row_slot.data(),token.data(),pos.data(),out.data()); });
    const auto b=Clock::now();
    if (e) return e;
    if ((e=api("h5_accept",[&] {
      for (int s=0;s<streams;s++) {
        h5_accept(ptr,s,accept+1); ingest[s]=accept+1;pos0[s]=base;
        anchor[s]=fixed_token(offset+s,base+accept+1);start[s]=base+accept+1;
      }
      return 0;
    }))) return e;
    round_input_hash=hash_ints(start,hash_ints(anchor,hash_ints(pos0,hash_ints(row0,hash_ints(ingest,round_input_hash)))));
    if ((e=api("h5_draft",[&] { return h5_draft(ptr,streams,slots.data(),ingest.data(),row0.data(),pos0.data(),anchor.data(),start.data(),drafts.data()); }))) return e;
    const auto c=Clock::now();
    verify_ms+=ms(a,b); draft_ms+=ms(b,c);
    if (record) {
      observation.verify.insert(observation.verify.end(),out.begin(),out.end());
      observation.draft.insert(observation.draft.end(),drafts.begin(),drafts.end());
      const float* conf=h5_draft_conf(ptr);
      if (!conf) return {Failure::Api,"h5_draft_conf",-1,cudaSuccess,"null confidence buffer"};
      for (int i=0;i<streams*7;i++) { unsigned bits; std::memcpy(&bits,conf+i,sizeof(bits)); observation.confidence.push_back(bits); }
    }
    if (diagnostic) return draft_checkpoint(iteration);
    return {};
  }
};

enum class Arm { AB, BA, Concurrent };
static const char* arm_name(Arm a) {
  switch (a) { case Arm::AB:return "serial_ab"; case Arm::BA:return "serial_ba"; case Arm::Concurrent:return "concurrent"; }
  std::abort();
}
// Persistent worker threads are used for every arm. The mutex release is the concurrent
// common-start barrier; final completion joins both engine streams through the public calls.
class Workers {
  std::array<Engine*,2> engines;
  std::array<std::thread,2> threads;
  std::mutex mu; std::condition_variable cv;
  std::array<bool,2> ready{{false,false}}, request{{false,false}}, done{{false,false}};
  std::array<Error,2> init, result;
  bool stop=false, record=false;
  int iteration=0;
public:
  Workers(Engine& a, Engine& b): engines{{&a,&b}} {}
  Error launch(int device) {
    try {
    for (int id=0;id<2;id++) threads[id]=std::thread([&,id,device] {
      const Error startup=cuda_error(cudaSetDevice(device),"worker cudaSetDevice");
      std::unique_lock<std::mutex> lock(mu); init[id]=startup;ready[id]=true;cv.notify_all();
      for (;;) {
        cv.wait(lock,[&] { return stop || request[id]; });
        if (stop) return;
        const int it=iteration;const bool save=record;request[id]=false;
        lock.unlock();const Error e=startup ? startup : engines[id]->round(it,save);lock.lock();
        result[id]=e;done[id]=true;cv.notify_all();
      }
    });
    } catch (const std::exception& ex) {
      return {Failure::Foreign,"worker thread creation",0,cudaSuccess,ex.what()};
    }
    std::unique_lock<std::mutex> lock(mu);cv.wait(lock,[&] { return ready[0] && ready[1]; });
    return init[0] ? init[0] : init[1];
  }
  ~Workers() {
    { std::lock_guard<std::mutex> lock(mu);stop=true;cv.notify_all(); }
    for (auto& t:threads) if (t.joinable()) t.join();
  }
  std::array<Error,2> run(Arm arm,int it,bool save,double& wall) {
    std::unique_lock<std::mutex> lock(mu);iteration=it;record=save;done={{false,false}};
    const auto begin=Clock::now();
    if (arm==Arm::Concurrent) {
      request={{true,true}};cv.notify_all();cv.wait(lock,[&] { return done[0] && done[1]; });
    } else {
      const int first=arm==Arm::AB ? 0 : 1,second=1-first;
      request[first]=true;cv.notify_all();cv.wait(lock,[&] { return done[first]; });
      request[second]=true;cv.notify_all();cv.wait(lock,[&] { return done[second]; });
    }
    wall=ms(begin,Clock::now());return result;
  }
};
static Error memory(const char* label) {
  size_t free,total;
  auto e=cuda_error(cudaMemGetInfo(&free,&total),"cudaMemGetInfo");
  if (e) return e;
  std::printf("{\"event\":\"memory\",\"label\":%s,\"free_bytes\":%zu,\"total_bytes\":%zu,\"used_bytes\":%zu}\n",quote(label).c_str(),free,total,total-free);
  std::fflush(stdout);return {};
}
static Observation joined(const Engine& a,const Engine& b,int rounds) {
  Observation o;
  for (int it=0;it<rounds;it++) {
    auto append=[&](const auto& v,auto& dst,int count) { dst.insert(dst.end(),v.begin()+it*count,v.begin()+(it+1)*count); };
    append(a.observation.verify,o.verify,32*8);append(b.observation.verify,o.verify,32*8);
    append(a.observation.draft,o.draft,32*7);append(b.observation.draft,o.draft,32*7);
    append(a.observation.confidence,o.confidence,32*7);append(b.observation.confidence,o.confidence,32*7);
  }
  return o;
}
static Error save(const std::string& path,const Observation& o,int rounds) {
  std::ofstream f(path);
  if (!f) return {Failure::Io,"open output",0,cudaSuccess,path};
  f << "h5-multi-engine-v1 " << rounds << "\n";
  for (int it=0;it<rounds;it++) for (int s=0;s<64;s++) {
    f << it << ' ' << s;
    for (int k=0;k<8;k++) f << ' ' << o.verify[(it*64+s)*8+k];
    for (int k=0;k<7;k++) f << ' ' << o.draft[(it*64+s)*7+k];
    for (int k=0;k<7;k++) f << ' ' << o.confidence[(it*64+s)*7+k];
    f << '\n';
  }
  f.close();
  if (!f) return {Failure::Io,"write output",0,cudaSuccess,path};
  return {};
}
struct Difference { size_t count=0; long first=-1; };
template<class T> static Difference difference(const std::vector<T>& a,const std::vector<T>& b) {
  Difference d;
  for (size_t i=0;i<std::max(a.size(),b.size());i++) if (i>=a.size() || i>=b.size() || a[i]!=b[i]) {
    d.count++;if (d.first<0) d.first=long(i);
  }
  return d;
}
static bool compare(const Observation& baseline,const Observation& now,const char* arm,int repeat) {
  const auto v=difference(baseline.verify,now.verify),d=difference(baseline.draft,now.draft),c=difference(baseline.confidence,now.confidence);
  std::printf("{\"event\":\"equivalence\",\"arm\":%s,\"repeat\":%d,\"verify_differences\":%zu,\"verify_first\":%ld,\"draft_differences\":%zu,\"draft_first\":%ld,\"confidence_bit_differences\":%zu,\"confidence_first\":%ld,\"exact\":%s}\n",
    quote(arm).c_str(),repeat,v.count,v.first,d.count,d.first,c.count,c.first,(v.count+d.count+c.count)==0 ? "true" : "false");
  return v.count+d.count+c.count==0;
}
static void timing(const char* arm,int repeat,const std::vector<double>& values,const Engine& a,const Engine* b) {
  auto sorted=values;std::sort(sorted.begin(),sorted.end());double total=0;for (double v:values) total+=v;
  std::printf("{\"event\":\"timing\",\"arm\":%s,\"repeat\":%d,\"rounds\":%zu,\"wall_round_mean_ms\":%.6f,\"wall_round_min_ms\":%.6f,\"wall_round_median_ms\":%.6f,\"wall_round_max_ms\":%.6f,\"engine0_verify_mean_ms\":%.6f,\"engine0_draft_mean_ms\":%.6f,\"engine1_verify_mean_ms\":%.6f,\"engine1_draft_mean_ms\":%.6f}\n",
    quote(arm).c_str(),repeat,values.size(),total/values.size(),sorted.front(),sorted[sorted.size()/2],sorted.back(),a.verify_ms/values.size(),a.draft_ms/values.size(),b ? b->verify_ms/values.size() : 0,b ? b->draft_ms/values.size() : 0);
}
static Error checked_restore(Engine& e,const char* arm,int sequence) { return e.restore(arm,sequence); }
int main(int argc,char** argv) {
  Config c;
  const std::array<const char*,8> modes{{"pair","single64","diagnose-pair","diagnose-single64","fresh-ab","fresh-ba","fresh-concurrent","fresh-single64"}};
  const bool known=argc==13 && std::any_of(modes.begin(),modes.end(),[&](const char* mode) { return !std::strcmp(argv[1],mode); });
  if (!known ||
      !integer(argv[4],8,8192,c.context) || c.context%8 || !integer(argv[5],1,100,c.rounds) ||
      !integer(argv[6],0,7,c.accept) || !integer(argv[7],1,10,c.repeats) ||
      !integer(argv[8],0,63,c.device) || (std::strcmp(argv[11],"graphs") && std::strcmp(argv[11],"eager"))) {
    std::fprintf(stderr,"usage: multi-engine-batch MODE MODEL DRAFTER CTX ROUNDS ACCEPT REPEATS DEVICE PLAN OUTPUT_PREFIX graphs|eager LABEL\nMODE: pair, single64, diagnose-pair, diagnose-single64, fresh-ab, fresh-ba, fresh-concurrent, fresh-single64\nCTX must be a multiple of 8; fresh modes require REPEATS=1 and have no warmup\n");return 2;
  }
  const std::string mode=argv[1];
  c.single=mode=="single64" || mode=="diagnose-single64" || mode=="fresh-single64";
  c.fresh=mode.rfind("fresh-",0)==0;c.diagnostic=c.fresh || mode.rfind("diagnose-",0)==0;
  c.fresh_arm=c.fresh ? mode.substr(6) : "";
  if (c.fresh && c.repeats!=1) { std::fprintf(stderr,"fresh modes require REPEATS=1\n");return 2; }
  c.model=argv[2];c.drafter=argv[3];c.plan=argv[9];c.output=argv[10];c.graph=argv[11];
  if (c.plan.empty() || !std::ifstream(c.plan)) { std::fprintf(stderr,"PLAN must name a readable frozen plan\n");return 2; }
  const char* reject[]={"H5_PLAN_WRITE","H5_PLAN_OVERRIDE","H5_SK_OVERRIDE","H5_PROF","H5_SYNC_DEBUG","H5_AB_PLAN_B","H5_NO_SPEC"};
  for (const char* key:reject) if (std::getenv(key)) { std::fprintf(stderr,"unset %s for this fixed-plan benchmark\n",key);return 2; }
  if (setenv("H5_PLAN_READ",c.plan.c_str(),1) || setenv("H5_PLAN_FILL","0",1) || setenv("H5_GRAPH_MAX_ROWS","512",1) ||
      (c.graph=="eager" ? setenv("H5_NOGRAPH","1",1) : unsetenv("H5_NOGRAPH"))) { std::perror("environment");return 1; }
  std::printf("{\"event\":\"config\",\"mode\":%s,\"label\":%s,\"model\":%s,\"drafter\":%s,\"plan\":%s,\"context\":%d,\"rounds\":%d,\"accept\":%d,\"repeats\":%d,\"device\":%d,\"graph\":%s,\"work\":\"64x8-fixed-inputs\",\"warmup_sequences\":%d,\"state_restore\":\"full-prefix-target-and-draft-ingest-replay\"}\n",
    quote(argv[1]).c_str(),quote(argv[12]).c_str(),quote(c.model).c_str(),quote(c.drafter).c_str(),quote(c.plan).c_str(),c.context,c.rounds,c.accept,c.repeats,c.device,quote(c.graph).c_str(),c.fresh ? 0 : 3);
  extern char** environ;
  for (char** p=environ;*p;p++) if (!std::strncmp(*p,"H5_",3)) std::printf("{\"event\":\"environment\",\"value\":%s}\n",quote(*p).c_str());
  Error e=cuda_error(cudaSetDevice(c.device),"main cudaSetDevice");
  if (e) { report(e,"setup",-1,-1);return 1; }
  if ((e=memory("before_engine0"))) { report(e,"setup",-1,-1);return 1; }
  Engine a(c.single ? 64 : 32,0,c);
  if ((e=a.load(c))) { report(e,"load",0,-1);const auto m=memory("engine0_load_failed");if (m) report(m,"memory",0,-1);return 1; }
  if ((e=memory("after_engine0"))) { report(e,"setup",0,-1);return 1; }
  if (c.single) {
    Observation reference;bool exact=true;
    for (int seq=c.fresh ? 0 : -3;seq<c.repeats;seq++) {
      if ((e=checked_restore(a,"single64",seq))) { report(e,"single64_restore",0,seq);return 1; }
      std::vector<double> times;
      for (int it=0;it<c.rounds;it++) { auto t=Clock::now();e=a.round(it,seq>=0);times.push_back(ms(t,Clock::now()));if (e) { report(e,"single64",0,it);return 1; } }
      if (seq>=0) {
        timing("single64",seq,times,a,nullptr);
        if ((e=save(c.output+".single64."+std::to_string(seq)+".txt",a.observation,c.rounds))) { report(e,"save",0,seq);return 1; }
        if (seq==0) reference=a.observation;else exact=compare(reference,a.observation,"single64",seq)&&exact;
      }
    }
    if ((e=memory("after_single64_graphs"))) { report(e,"memory",0,-1);return 1; }
    if (c.fresh) std::printf("{\"event\":\"fresh_result\",\"comparison\":\"not-run\",\"timing\":\"first-sequence-not-steady-state\"}\n");
    if (c.diagnostic && !c.fresh) std::printf("{\"event\":\"prefix_result\",\"exact\":%s}\n",a.prefix_exact ? "true" : "false");
    return exact && a.prefix_exact ? 0 : 3;
  }
  Engine b(32,32,c);
  if ((e=b.load(c))) { report(e,"load",1,-1);const auto m=memory("engine1_load_failed");if (m) report(m,"memory",1,-1);return 1; }
  if ((e=memory("after_engine1"))) { report(e,"setup",1,-1);return 1; }
  Workers workers(a,b);
  if ((e=workers.launch(c.device))) { report(e,"worker_start",-1,-1);return 1; }
  for (int seq=0;seq<(c.fresh ? 0 : 3);seq++) {
    if ((e=checked_restore(a,"warmup",seq))) { report(e,"warmup_restore",0,seq);return 1; }
    if ((e=checked_restore(b,"warmup",seq))) { report(e,"warmup_restore",1,seq);return 1; }
    for (int it=0;it<c.rounds;it++) {
      double wall;const auto errors=workers.run(Arm::AB,it,false,wall);
      for (int id=0;id<2;id++) if (errors[id]) { report(errors[id],"warmup",id,it);return 1; }
    }
  }
  if ((e=memory("after_pair_graphs"))) { report(e,"setup",-1,-1);return 1; }
  Observation baseline;bool exact=true;
  std::array<double,3> wall_total{{0,0,0}};
  const std::array<Arm,3> arms{{Arm::AB,Arm::BA,Arm::Concurrent}};
  for (int repeat=0;repeat<c.repeats;repeat++) for (int slot=0;slot<(c.fresh ? 1 : 3);slot++) {
    // Rotate arm order between repetitions; the first baseline is always serial AB.
    const Arm arm=c.fresh ? (c.fresh_arm=="ab" ? Arm::AB : c.fresh_arm=="ba" ? Arm::BA : Arm::Concurrent) : arms[(slot+repeat)%3];
    const char* name=arm_name(arm);
    if ((e=checked_restore(a,name,repeat))) { report(e,"arm_restore",0,repeat);return 1; }
    if ((e=checked_restore(b,name,repeat))) { report(e,"arm_restore",1,repeat);return 1; }
    std::vector<double> times;
    for (int it=0;it<c.rounds;it++) {
      double wall;const auto errors=workers.run(arm,it,true,wall);times.push_back(wall);wall_total[size_t(arm)]+=wall;
      for (int id=0;id<2;id++) if (errors[id]) { report(errors[id],name,id,it);return 1; }
      std::printf("{\"event\":\"round\",\"arm\":%s,\"repeat\":%d,\"iteration\":%d,\"wall_ms\":%.6f,\"engine0_input_hash\":%s,\"engine1_input_hash\":%s}\n",quote(name).c_str(),repeat,it,wall,quote(std::to_string(a.round_input_hash)).c_str(),quote(std::to_string(b.round_input_hash)).c_str());
    }
    timing(name,repeat,times,a,&b);Observation now=joined(a,b,c.rounds);
    if ((e=save(c.output+"."+name+"."+std::to_string(repeat)+".txt",now,c.rounds))) { report(e,"save",-1,repeat);return 1; }
    if (repeat==0 && slot==0) baseline=now;
    else exact=compare(baseline,now,name,repeat)&&exact;
    std::fflush(stdout);
  }
  if (c.fresh) { std::printf("{\"event\":\"fresh_result\",\"comparison\":\"not-run\",\"timing\":\"first-sequence-not-steady-state\"}\n");return 0; }
  const bool prefix_exact=a.prefix_exact && b.prefix_exact;
  if (c.diagnostic) std::printf("{\"event\":\"prefix_result\",\"exact\":%s}\n",prefix_exact ? "true" : "false");
  const double count=c.rounds*c.repeats;
  std::printf("{\"event\":\"result\",\"same_geometry_exact\":%s,\"serial_ab_mean_ms\":%.6f,\"serial_ba_mean_ms\":%.6f,\"concurrent_mean_ms\":%.6f,\"ab_over_concurrent\":%.6f,\"ba_over_concurrent\":%.6f,\"ab_saved_us_per_round\":%.6f,\"ba_saved_us_per_round\":%.6f}\n",
    exact ? "true" : "false",wall_total[0]/count,wall_total[1]/count,wall_total[2]/count,
    wall_total[0]/wall_total[2],wall_total[1]/wall_total[2],
    (wall_total[0]-wall_total[2])*1000/count,(wall_total[1]-wall_total[2])*1000/count);
  return exact && prefix_exact ? 0 : 3;
}
