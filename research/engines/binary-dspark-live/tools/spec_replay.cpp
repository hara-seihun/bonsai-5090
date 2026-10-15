// Single-stream speculative decoding loop, in process: what the server runs for one request in
// strict native7 (eight-row verifier round, then the DFlash2 drafter ingesting the accepted rows
// and drafting the next block). Real greedy acceptance, no tokenizer: the prompt is the first
// STEP of a recorded replay event file.
//
//   spec-replay MODEL DRAFTER EVENT_FILE OUT_PREFIX --rounds N --passes P
//
// Each pass: reset slot 0, prefill the prompt (one step + drafter ingest), then up to N rounds or
// until the verifier emits EOS. Warm-up: three unrecorded passes. Measured passes alternate arms in
// ABBA order when H5_AB_PREWAIT=A,B, H5_AB_L2PF=A,B or H5_AB_VARIANT=A,B (hex) is set (same knobs as profile-replay),
// or H5_AB_ENV='NAME=A|B': two engines (each with its own drafter and graphs), NAME set to A or B while
// that engine is created, warmed, captured and measured. Only for knobs read at launch/capture time.
// OUT_PREFIX.csv: pass,arm,round,accepted,verify_s,host_s,draft_s,round_s
// H5_SMLIT builds: H5_SMLIT_ROUNDS=FIRST:END records measured pass 0 rounds [FIRST,END), with two
// bounds per round (after the verifier step, after the drafter), so OUT_PREFIX-smlit.bin segments
// alternate verify, draft. tools/smlit_spec.py splits them and measures the gaps between.
#ifdef H5_SMLIT
#include "smlit.cuh"
extern "C" void h5_lit_set_kernels(h5::lit::Buf*);extern "C" void h5_lit_set_fp4(h5::lit::Buf*);
extern "C" void h5_lit_set_draft(h5::lit::Buf*);extern "C" void h5_lit_set_gmem(h5::lit::Buf*);extern "C" void h5_lit_set_sk(h5::lit::Buf*);
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "head_demand.h"
#include "sampling_contract.h"
extern "C" {
void* h5_create_draft(const char*,int,int,int,int); int h5_draft_load(void*,const char*);
void h5_pretune(void*); void h5_capture(void*,int);
void h5_reset(void*,int); void h5_keep(void*,int,int); void h5_accept(void*,int,int);
int h5_draft(void*,int,const int*,const int*,const int*,const int*,const int*,const int*,int*);
void h5_set_prewait_cap(int); void h5_set_l2pf_stages(int); void h5_set_prep_wide(int);
void h5_set_variant_kernels(unsigned); void h5_set_variant_fp4(unsigned); void h5_set_variant_draft(unsigned);
}
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }
static bool eos(int t) { return t == 248046 || t == 248044; }

int main(int argc, char** argv) { try {
  if (argc < 5) throw std::runtime_error("usage: spec-replay MODEL DRAFTER EVENT_FILE OUT_PREFIX --rounds N --passes P [--rows 8|16] [--sampled]");
  int rounds = -1, passes = -1, verify_rows = 8;
  bool sampled = false;
  H5SamplingConfig target_sampling{1.0, .95, .05, 0.0, 1.0, 20};
  H5SamplingConfig draft_sampling = target_sampling; draft_sampling.temperature *= .5;
  constexpr uint64_t sample_seed = 12345;
  for (int i = 5; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--rounds" && rounds < 0 && i + 1 < argc) rounds = std::stoi(argv[++i]);
    else if (a == "--passes" && passes < 0 && i + 1 < argc) passes = std::stoi(argv[++i]);
    else if (a == "--rows" && i + 1 < argc) { verify_rows = std::stoi(argv[++i]); if (verify_rows != 8 && verify_rows != 16) throw std::runtime_error("--rows requires 8 or 16"); }
    else if (a == "--sampled" && !sampled) sampled = true;
    else throw std::runtime_error("unknown or repeated option: " + a);
  }
  if (rounds < 1 || rounds > 480) throw std::runtime_error("--rounds must be 1..480");
  if (passes < 1 || passes > 100) throw std::runtime_error("--passes must be 1..100");
  std::vector<int> prompt;
  { std::ifstream in(argv[3]); if (!in) throw std::runtime_error("event file missing"); std::string line;
    while (std::getline(in, line)) { std::istringstream s(line); std::string op; s >> op; if (op != "STEP") continue;
      int n; s >> n; for (int i = 0; i < n; i++) { int sl, tok, pos; if (!(s >> sl >> tok >> pos) || sl != 0 || pos != i) throw std::runtime_error("first STEP must be a slot-0 prefill from position 0"); prompt.push_back(tok); }
      break; } }
  if (prompt.empty() || prompt.size() > 512) throw std::runtime_error("prompt missing or longer than one prefill step");
  const int np = prompt.size();
  struct EnvArm { std::string name, value[2]; };
  std::vector<EnvArm> env_arms;
  if (const char* v = std::getenv("H5_AB_ENV")) {
    std::istringstream entries(v); std::string x;
    while (std::getline(entries, x, ';')) {
      auto eq = x.find('='), bar = x.find('|');
      if (eq == std::string::npos || bar == std::string::npos || bar < eq || eq == 0 || x.find('|', bar + 1) != std::string::npos)
        throw std::runtime_error("H5_AB_ENV must be NAME=A|B[;NAME=A|B...]");
      EnvArm entry{x.substr(0, eq), {x.substr(eq + 1, bar - eq - 1), x.substr(bar + 1)}};
      for (const auto& old : env_arms) if (old.name == entry.name) throw std::runtime_error("duplicate H5_AB_ENV name");
      env_arms.push_back(entry);
    }
    if (env_arms.empty() || !*v || v[std::strlen(v)-1] == ';') throw std::runtime_error("empty H5_AB_ENV assignment");
  }
  auto set_arm_env = [&](int arm) { for (const auto& entry : env_arms) setenv(entry.name.c_str(), entry.value[arm].c_str(), 1); };
  // H5_AB_PREPWIDE=A,B: two engines, wide row prep off/on (process-wide setter applied per arm).
  int pw[2] = {-1, -1};
  if (const char* v = std::getenv("H5_AB_PREPWIDE")) { if (std::sscanf(v, "%d,%d", &pw[0], &pw[1]) != 2 || pw[0] < 0 || pw[0] > 2 || pw[1] < 0 || pw[1] > 2 || !env_arms.empty()) throw std::runtime_error("H5_AB_PREPWIDE must be A,B in 0..2 (2 = multi-row only), alone");
    env_arms.push_back({"H5_AB_PREPWIDE_ARM", {std::to_string(pw[0]), std::to_string(pw[1])}}); }
  auto make = [&](int arm) {
    set_arm_env(arm);
    if (pw[0] >= 0) h5_set_prep_wide(pw[arm]);
    void* x = h5_create_draft(argv[1], 1, 4096, 520, verify_rows - 1); if (!x) throw std::runtime_error("model load failed");
    if (h5_draft_load(x, argv[2])) throw std::runtime_error("drafter load failed");
    h5_capture(x, 1); h5_pretune(x); return x; };
  void* engines[2] = {make(0), env_arms.empty() ? nullptr : make(1)};
  void* e = engines[0];

  std::vector<int> ab; std::vector<unsigned> abv;
  if (const char* v = std::getenv("H5_AB_PREWAIT")) { int x, y; if (std::sscanf(v, "%d,%d", &x, &y) != 2 || x < 0 || y < 0) throw std::runtime_error("H5_AB_PREWAIT must be A,B"); ab = {x, y}; }
  std::vector<int> abl;
  if (const char* v = std::getenv("H5_AB_L2PF")) { int x, y; if (std::sscanf(v, "%d,%d", &x, &y) != 2 || x < 0 || y < 0 || x > 64 || y > 64) throw std::runtime_error("H5_AB_L2PF must be A,B in 0..64"); abl = {x, y}; }
  if (const char* v = std::getenv("H5_AB_VARIANT")) { unsigned x, y; if (std::sscanf(v, "%x,%x", &x, &y) != 2) throw std::runtime_error("H5_AB_VARIANT must be A,B hex"); abv = {x, y}; }
  if ((!ab.empty()) + (!abv.empty()) + (!abl.empty()) + (!env_arms.empty()) > 1) throw std::runtime_error("one A/B mechanism at a time");
  const bool armed = !ab.empty() || !abv.empty() || !abl.empty() || !env_arms.empty();
  auto select = [&](int arm) { if (env_arms.empty()) return; set_arm_env(arm); if (pw[0] >= 0) h5_set_prep_wide(pw[arm]); e = engines[arm]; };

#ifdef H5_SMLIT
  const char* lit_env = std::getenv("H5_SMLIT_ROUNDS"); if (!lit_env) throw std::runtime_error("H5_SMLIT build requires H5_SMLIT_ROUNDS=FIRST:END");
  unsigned lit_first = 0, lit_end = 0; if (std::sscanf(lit_env, "%u:%u", &lit_first, &lit_end) != 2 || lit_end <= lit_first || lit_end > (unsigned)rounds) throw std::runtime_error("invalid H5_SMLIT_ROUNDS");
  const unsigned long long lit_cap = (unsigned long long)(lit_end - lit_first) * 1800000ULL;
  h5::lit::Buf* lit_dev = nullptr; if (cudaMalloc(&lit_dev, sizeof(h5::lit::Buf) + lit_cap * sizeof(h5::lit::Rec)) != cudaSuccess) throw std::runtime_error("smlit allocation failed");
  { h5::lit::Buf head{}; head.count = 0; head.cap = lit_cap; if (cudaMemcpy(lit_dev, &head, 16, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("smlit init failed"); }
  std::vector<unsigned long long> lit_bounds;
  auto lit_set = [&](h5::lit::Buf* b) { h5_lit_set_kernels(b); h5_lit_set_fp4(b); h5_lit_set_draft(b); h5_lit_set_gmem(b); h5_lit_set_sk(b); };
  auto lit_mark = [&]() { unsigned long long c = 0; if (cudaMemcpy(&c, lit_dev, 8, cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("smlit count read failed"); lit_bounds.push_back(c); };
#endif

  struct Row { int pass, arm, round, accepted; double verify, sample, host, draft, total; };
  std::vector<Row> rows; std::vector<double> prefill_s; std::vector<std::vector<int>> outputs;
  auto run = [&](int pass, int arm, bool record) {
    const bool lit_pass = record && pass == 0;
    h5_reset(e, 0); h5_keep(e, 0, 0);
    std::vector<int> sl(np, 0), pos(np), out(np), head_rows, gen;
    for (int i = 0; i < np; i++) pos[i] = i;
    h5::append_head_rows(head_rows, 0, np, h5::HeadUse::FinishingPrefill);
    auto t0 = clk::now();
    if (h5::step_head_rows(e, np, sl.data(), prompt.data(), pos.data(), out.data(), H5_HEAD_SELECTED, head_rows)) throw std::runtime_error("prefill step failed");
    std::vector<unsigned char> seen(248320, 0);
    if (sampled && h5_sample_target(e, &target_sampling, sample_seed, 0, seen.data(), nullptr, 0, 1, out.data() + np - 1))
      throw std::runtime_error("prefill sample failed");
    int len = np, anchor = out[np - 1]; std::vector<int> drafts(verify_rows - 1);
    gen.push_back(anchor); seen[anchor] = 1;
    auto draft = [&](int ni, int ip, int an) {
      int s = 0, ir = 0, st = len;
      return sampled ? h5_draft_sampled(e, 1, &s, &ni, &ir, &ip, &an, &st, drafts.data(), &draft_sampling, sample_seed, gen.size(), seen.data())
                     : h5_draft(e, 1, &s, &ni, &ir, &ip, &an, &st, drafts.data());
    };
    if (draft(np, 0, anchor)) throw std::runtime_error("prefill draft failed");
    if (record) prefill_s.push_back(secs(t0, clk::now()));
    std::vector<int> rs(verify_rows, 0), rt(verify_rows), rp(verify_rows), ro(verify_rows);
    head_rows.clear(); h5::append_head_rows(head_rows, 0, verify_rows, h5::HeadUse::Verify);
    for (int r = 0; r < rounds && !eos(anchor) && len + verify_rows + 2 < 4096; r++) {
#ifdef H5_SMLIT
      if (lit_pass && (unsigned)r == lit_first) lit_set(lit_dev);
#endif
      auto a = clk::now();
      h5_keep(e, 0, 1);
      rt[0] = anchor; rp[0] = len;
      for (int k = 0; k < verify_rows - 1; k++) { rt[k + 1] = drafts[k]; rp[k + 1] = len + 1 + k; }
      if (h5::step_head_rows(e, verify_rows, rs.data(), rt.data(), rp.data(), ro.data(), H5_HEAD_SELECTED, head_rows)) throw std::runtime_error("verify step failed");
      auto b = clk::now();
      if (sampled && h5_sample_target(e, &target_sampling, sample_seed, gen.size(), seen.data(), drafts.data(), 0, verify_rows, ro.data()))
        throw std::runtime_error("verify sample failed");
      auto sampled_at = clk::now();
#ifdef H5_SMLIT
      if (lit_pass && (unsigned)r >= lit_first && (unsigned)r < lit_end) lit_mark();
#endif
      int acc = 0; while (acc < verify_rows - 1 && ro[acc] == drafts[acc]) acc++;
      for (int k = 0; k < acc; k++) if (eos(ro[k])) { acc = k; break; }
      h5_accept(e, 0, acc + 1);
      for (int k = 0; k <= acc; k++) { gen.push_back(ro[k]); seen[ro[k]] = 1; }
      const int ip = len; len += acc + 1; anchor = ro[acc];
      int an = eos(anchor) ? -1 : anchor;
      auto c = clk::now();
      if (draft(acc + 1, ip, an)) throw std::runtime_error("draft failed");
      auto d = clk::now();
#ifdef H5_SMLIT
      if (lit_pass && (unsigned)r >= lit_first && (unsigned)r < lit_end) { lit_mark(); if ((unsigned)r + 1 == lit_end) lit_set(nullptr); }
#endif
      if (record) rows.push_back({pass, arm, r, acc, secs(a, b), secs(b, sampled_at), secs(sampled_at, c), secs(c, d), secs(a, d)});
    }
    if (record) outputs.push_back(gen);
  };
  for (int arm = 0; arm < (env_arms.empty() ? 1 : 2); arm++) { select(arm); for (int w = 0; w < 3; w++) run(-1, arm, false); }
  select(0);
  std::string arms;
  for (int p = 0; p < passes; p++) {
    const int arm = ((p & 3) == 0 || (p & 3) == 3) ? 0 : 1;
    if (!ab.empty()) h5_set_prewait_cap(ab[arm]);
    if (!abl.empty()) h5_set_l2pf_stages(abl[arm]);
    if (!abv.empty()) { h5_set_variant_kernels(abv[arm]); h5_set_variant_fp4(abv[arm]); h5_set_variant_draft(abv[arm]); }
    select(arm);
    if (armed) arms += char('A' + arm);
    run(p, armed ? arm : 0, true);
  }
#ifdef H5_SMLIT
  { if (lit_bounds.size() != 2 * (lit_end - lit_first)) throw std::runtime_error("smlit window not reached (EOS before END?)");
    unsigned long long n = lit_bounds.back(); if (n > lit_cap) throw std::runtime_error("smlit overflow: " + std::to_string(n));
    std::vector<h5::lit::Rec> recs(n); if (n && cudaMemcpy(recs.data(), lit_dev->recs, n * sizeof(h5::lit::Rec), cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("smlit read failed");
    std::ofstream f(std::string(argv[4]) + "-smlit.bin", std::ios::binary);
    unsigned long long hdr[3] = {0x544c4d53ULL, (unsigned long long)lit_first, (unsigned long long)lit_bounds.size()};
    f.write((const char*)hdr, sizeof(hdr)); f.write((const char*)lit_bounds.data(), lit_bounds.size() * 8); f.write((const char*)recs.data(), n * sizeof(h5::lit::Rec));
    if (!f) throw std::runtime_error("smlit write failed"); }
#endif
  { std::ofstream o(std::string(argv[4]) + ".csv"); if (!o) throw std::runtime_error("csv open failed"); o << std::setprecision(9) << "pass,arm,round,accepted,verify_s,sample_s,host_s,draft_s,round_s\n";
    for (auto& r : rows) o << r.pass << "," << r.arm << "," << r.round << "," << r.accepted << "," << r.verify << "," << r.sample << "," << r.host << "," << r.draft << "," << r.total << "\n";
    if (!o) throw std::runtime_error("csv write failed"); }
  { std::ofstream o(std::string(argv[4]) + "-tokens.txt"); for (auto& g : outputs) { for (int t : g) o << t << " "; o << "\n"; } }
  if (!arms.empty()) std::cerr << "ab_arms " << arms << "\n";
  double pre = 0; for (double x : prefill_s) pre += x;
  std::cout << "{\"verify_rows\":" << verify_rows << ",\"sampled\":" << (sampled ? "true" : "false") << ",\"passes\":" << passes << ",\"rounds_recorded\":" << rows.size() << ",\"prefill_mean_s\":" << pre / prefill_s.size() << "}\n";
  return 0;
} catch (const std::exception& ex) { std::cerr << ex.what() << "\n"; return 1; } }
