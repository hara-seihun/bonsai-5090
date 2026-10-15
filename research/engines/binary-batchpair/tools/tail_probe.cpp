// Producer-tail prep probe: logits of the same steps with H5_TAIL_PREP masks against separate preps.
//
//   tail-probe MODEL EVENT_FILE [ROWS] [STEPS] [MASK...]
//
// Prefills the first STEP of EVENT_FILE (prompt; tail never applies above 8 rows), then runs STEPS
// steps of ROWS rows whose tokens are the prompt's own tokens (fixed inputs, identical in every arm),
// reading all logits (direct launches, H5_TAIL_PREP read per step). Arm "0" runs twice to show the
// split-K atomic noise floor. Prints per arm: top-1 agreement with the first "0" run, max |dlogit|,
// mean KL(ref || arm) over all rows of all steps.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
void* h5_create(const char*, int, int, int);
void h5_pretune(void*); void h5_reset(void*, int);
int h5_step(void*, int, const int*, const int*, const int*, int*);
int h5_step_logits(void*, int, const int*, const int*, const int*, int*, float*);
int h5_tail_dbg(void*, unsigned long long*);
void h5_set_variant_kernels(unsigned); void h5_set_variant_fp4(unsigned);
}
constexpr int VOCAB = 248320;
int main(int argc, char** argv) { try {
  if (argc < 3) throw std::runtime_error("usage: tail-probe MODEL EVENT_FILE [ROWS] [STEPS] [MASK...]");
  const int R = argc > 3 ? atoi(argv[3]) : 8, S = argc > 4 ? atoi(argv[4]) : 6;
  if (R < 1 || R > 8 || S < 1 || S > 64) throw std::runtime_error("ROWS must be 1..8, STEPS 1..64");
  std::vector<std::string> arms = {"0", "0"};
  for (int i = 5; i < argc; i++) arms.push_back(argv[i]);
  if (argc <= 5) for (const char* m : {"15", "1", "2", "4", "8"}) arms.push_back(m);
  std::vector<int> prompt;
  { std::ifstream in(argv[2]); std::string line;
    while (std::getline(in, line)) { std::istringstream s(line); std::string op; s >> op; if (op != "STEP") continue;
      int n; s >> n; for (int i = 0; i < n; i++) { int sl, tok, pos; s >> sl >> tok >> pos; prompt.push_back(tok); } break; } }
  const int np = prompt.size();
  if (np < R * S || np > 512) throw std::runtime_error("prompt must hold ROWS*STEPS tokens and fit one prefill step");
  void* e = h5_create(argv[1], 1, 4096, 520); if (!e) throw std::runtime_error("model load failed");
  h5_pretune(e);
  if (const char* v = getenv("PROBE_VARIANT")) { const unsigned x = (unsigned)strtoul(v, nullptr, 16); h5_set_variant_kernels(x); h5_set_variant_fp4(x); }
  std::vector<int> slot(np, 0), pos(np), out(np);
  for (int i = 0; i < np; i++) pos[i] = i;
  std::vector<std::vector<float>> ref;
  for (size_t a = 0; a < arms.size(); a++) {
    static const char* knob = getenv("PROBE_ENV") ? getenv("PROBE_ENV") : "H5_TAIL_PREP";
    if (getenv("PROBE_VARIANTS")) {   // arms are hex variant words (device bits), not values of PROBE_ENV
      const unsigned x = (unsigned)strtoul(arms[a].c_str(), nullptr, 16); h5_set_variant_kernels(x); h5_set_variant_fp4(x);
    } else setenv(knob, arms[a].c_str(), 1);
    h5_reset(e, 0);
    if (h5_step(e, np, slot.data(), prompt.data(), pos.data(), out.data())) throw std::runtime_error("prefill failed");
    std::vector<std::vector<float>> lg(S, std::vector<float>((size_t)R * VOCAB));
    for (int s = 0; s < S; s++) {
      std::vector<int> tk(prompt.begin() + s * R, prompt.begin() + (s + 1) * R), ps(R);
      for (int i = 0; i < R; i++) ps[i] = np + s * R + i;
      if (h5_step_logits(e, R, slot.data(), tk.data(), ps.data(), out.data(), lg[s].data())) throw std::runtime_error("step failed");
    }
    if (a == 0) { ref = lg; continue; }
    int agree = 0, n = 0; double maxd = 0, kl = 0; bool finite = true;
    for (int s = 0; s < S; s++) for (int r = 0; r < R; r++) {
      const float* p = ref[s].data() + (size_t)r * VOCAB; const float* q = lg[s].data() + (size_t)r * VOCAB;
      int ap = 0, aq = 0; double mp = p[0], mq = q[0];
      for (int v = 0; v < VOCAB; v++) {
        if (!std::isfinite(q[v])) finite = false;
        if (p[v] > p[ap]) ap = v; if (q[v] > q[aq]) aq = v;
        mp = std::max(mp, (double)p[v]); mq = std::max(mq, (double)q[v]);
        maxd = std::max(maxd, (double)std::fabs(p[v] - q[v]));
      }
      double zp = 0, zq = 0; for (int v = 0; v < VOCAB; v++) { zp += std::exp(p[v] - mp); zq += std::exp(q[v] - mq); }
      double k = 0; for (int v = 0; v < VOCAB; v++) { const double lp = p[v] - mp - std::log(zp), lq = q[v] - mq - std::log(zq); k += std::exp(lp) * (lp - lq); }
      kl += k; agree += ap == aq; n++;
    }
    printf("arm %s=%-3s top1 %d/%d  max|dlogit| %.4f  mean KL %.3e  %s\n", getenv("PROBE_ENV") ? getenv("PROBE_ENV") : "H5_TAIL_PREP", arms[a].c_str(), agree, n, maxd, kl / n, finite ? "finite" : "NONFINITE");
    fflush(stdout);
    std::vector<unsigned long long> dbg(8 * 32 * 8);
    if (h5_tail_dbg(e, dbg.data()) == 0) {
      const char* site[5] = {"gdn", "combine", "out", "gu", "down"};
      for (int k = 0; k < 5; k++) {
        unsigned long long t0 = ~0ull; int nc = 0;
        for (int c = 0; c < 32; c++) if (dbg[(k * 32 + c) * 8]) { t0 = std::min(t0, dbg[(k * 32 + c) * 8]); nc++; }
        if (!nc) continue;
        printf("  %-8s chunk: arrive@ +arrival +compute +rendezvous +quant (ns)\n", site[k]);
        for (int c = 0; c < 32; c++) {
          const unsigned long long* d = &dbg[(k * 32 + c) * 8];
          if (!d[0]) continue;
          printf("    %2d  %6llu  %5lld %5lld %5lld %5lld\n", c, d[0] - t0, (long long)(d[1] - d[0]), (long long)(d[2] - d[1]),
                 d[3] ? (long long)(d[3] - d[2]) : 0ll, (long long)(d[4] - (d[3] ? d[3] : d[2])));
        }
      }
    }
  }
  return 0;
} catch (std::exception& ex) { fprintf(stderr, "tail-probe: %s\n", ex.what()); return 1; } }
