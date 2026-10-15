// Batch-aware speculation: how many of each slot's proposals the verifier checks, given the load.
//
// At one or a few streams the verifier is bandwidth-bound and extra rows are nearly free; at many streams
// the FP4 GEMMs are tensor-core-bound and every verifier row costs about the same. Rows that are unlikely
// to be accepted then displace useful work. Modes (H5_SPEC_POLICY):
//   cut (default)  the existing rule: all proposals at <= H5_VERIFY_ALL_BELOW streams, else cut each block
//                  at the first selector gap below H5_DRAFT_CUT.
//   cap            cut as above, then at most cap(nact) proposals; H5_SPEC_CAP="N1:C1,N2:C2,..." means cap Ci
//                  once at least Ni streams are active (ascending Ni).
//   ev             expected-value selection over the whole step: proposal j of slot s is verified when the
//                  probability that it and all earlier proposals are accepted, q = prod_i<=j p(gap_i, i), is at
//                  least lambda * trow, where lambda is the step's own tokens/second ratio (Dinkelbach iteration)
//                  and trow the marginal seconds per verifier row. p(gap, j) = sigmoid(a + b*gap + c*j)
//                  (H5_SPEC_PCAL="a,b,c"); step cost t0 + tslot*slots + trow*max(rows, knee)
//                  (H5_SPEC_COST="t0_ms,tslot_ms,trow_ms,knee_rows").
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace h5::bspec {

enum class Mode { Cut, Cap, EV };
enum class Error { None, BadMode, BadCap, BadCal, BadCost };

struct Policy {
  Mode mode = Mode::Cut;
  std::vector<std::pair<int, int>> caps;   // (min active streams, cap), ascending
  float a = -0.93f, b = 1.155f, c = -0.03f;
  double t0 = 0, tslot = 0, trow = 0; int knee = 0;
};

inline const char* error(Error e) {
  switch (e) {
    case Error::None: return "ok";
    case Error::BadMode: return "H5_SPEC_POLICY must be cut, cap or ev";
    case Error::BadCap: return "H5_SPEC_CAP must be N:C[,N:C...] with ascending N >= 1 and 0 <= C <= 64";
    case Error::BadCal: return "H5_SPEC_PCAL must be three finite numbers a,b,c";
    case Error::BadCost: return "H5_SPEC_COST must be t0_ms,tslot_ms,trow_ms,knee_rows with trow > 0";
  }
  return "invalid bspec error";
}

inline Error configure(const char* mode, const char* cap, const char* cal, const char* cost, Policy& p) {
  p = Policy{};
  if (!mode || !std::string(mode).compare("cut")) p.mode = Mode::Cut;
  else if (!std::string(mode).compare("cap")) p.mode = Mode::Cap;
  else if (!std::string(mode).compare("ev")) p.mode = Mode::EV;
  else return Error::BadMode;
  if (p.mode == Mode::Cap) {
    if (!cap) return Error::BadCap;
    std::string s = cap; size_t i = 0; int last = 0;
    while (i < s.size()) {
      size_t j = std::min(s.find(',', i), s.size());
      int n = 0, cc = 0; char tail = 0;
      if (sscanf(s.substr(i, j - i).c_str(), "%d:%d%c", &n, &cc, &tail) != 2 || n <= last || cc < 0 || cc > 64) return Error::BadCap;
      p.caps.push_back({n, cc}); last = n; i = j + 1;
    }
    if (p.caps.empty()) return Error::BadCap;
  }
  if (p.mode == Mode::EV) {
    if (cal) {
      char tail = 0;
      if (sscanf(cal, "%f,%f,%f%c", &p.a, &p.b, &p.c, &tail) != 3 || !std::isfinite(p.a) || !std::isfinite(p.b) || !std::isfinite(p.c)) return Error::BadCal;
    }
    double t0 = 0, ts = 0, tr = 0; int knee = 0; char tail = 0;
    if (!cost || sscanf(cost, "%lf,%lf,%lf,%d%c", &t0, &ts, &tr, &knee, &tail) != 4 || !(tr > 0) || t0 < 0 || ts < 0 || knee < 0) return Error::BadCost;
    p.t0 = t0 * 1e-3; p.tslot = ts * 1e-3; p.trow = tr * 1e-3; p.knee = knee;
  }
  return Error::None;
}

inline int cap_for(const Policy& p, int nact) {
  int cap = 1 << 30;
  for (auto& [n, c] : p.caps) if (nact >= n) cap = c;
  return cap;
}

// EV selection. gaps: [n][P] selector gaps of each decode slot; extra_rows: prefill rows in the same step.
// Writes k[s] in [0, P]. Returns the modelled tokens/second of the chosen step.
inline double select_ev(const Policy& p, int n, int P, const float* gaps, int extra_rows, int* k) {
  if (n <= 0) return 0;
  std::vector<double> q((size_t)n * P);
  for (int s = 0; s < n; s++) {
    double run = 1;
    for (int j = 0; j < P; j++) {
      const double z = p.a + p.b * gaps[(size_t)s * P + j] + p.c * j;
      run *= 1.0 / (1.0 + std::exp(-z));
      q[(size_t)s * P + j] = run;
    }
  }
  auto evaluate = [&](double theta, double& tokens, int& rows) {
    tokens = n; rows = n + extra_rows;
    for (int s = 0; s < n; s++) {
      int kk = 0;
      while (kk < P && q[(size_t)s * P + kk] >= theta) { tokens += q[(size_t)s * P + kk]; kk++; }
      k[s] = kk; rows += kk;
    }
  };
  auto cost = [&](int rows) { return p.t0 + p.tslot * n + p.trow * std::max(rows, p.knee); };
  // Everything fits under the knee: rows are free, verify all.
  if (n * (P + 1) + extra_rows <= p.knee) {
    double tokens; int rows; evaluate(-1.0, tokens, rows); return tokens / cost(rows);
  }
  double lambda = 0, tokens = 0; int rows = 0;
  evaluate(-1.0, tokens, rows); lambda = tokens / cost(rows);
  for (int it = 0; it < 6; it++) {
    evaluate(lambda * p.trow, tokens, rows);
    const double next = tokens / cost(rows);
    if (std::fabs(next - lambda) <= 1e-6 * next) { lambda = next; break; }
    lambda = next;
  }
  evaluate(lambda * p.trow, tokens, rows);
  return tokens / cost(rows);
}

}  // namespace h5::bspec
