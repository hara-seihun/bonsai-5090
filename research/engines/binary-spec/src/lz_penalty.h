#pragma once
// LZ penalty (Ginart et al., "Lempel-Ziv Penalty", arXiv 2504.20131 v4, TMLR 2026) for speculative verify rows.
// See tools/ANTILOOP.md.
//
// The penalty simulates an LZ77/LZSS sliding-window compressor over the generated tokens and adds alpha times the
// codelength (bits) each candidate next token would cost to the logits: a token absent from the window costs a
// literal (log2 V), a token that occurs in the window costs a match at its distance (log2 delta), and a token that
// extends the longest current match costs almost nothing. Shifted so that a literal is 0, every adjustment is a
// non-positive penalty on a token present in the window, so a row needs at most WINDOW sparse entries.
//
// H5_LZ=ALPHA:WINDOW:BUFFER:FORM configures the server; ALPHA is the strength for requests that do not send
// "lz_penalty" (0 = such requests are unpenalized). A request's "lz_penalty" (0..4) sets its own strength.
// Forms:
//   paper  eq. 15 of v4: literal log2 V; extension of the longest suffix match (length l >= 1, copy distance d)
//          log2(1 + (d-l-1)/(l d)) - 1 with d raised to l+2 for overlapping (period < l) matches; any other
//          token in the window log2(nearest distance). The extension always pays the largest penalty,
//          alpha (log2 V + 1 - ...), whatever its distance.
//   dist   the authors' SGLang penalizer (tginart/sglang lz_penalty.py @ab08e697): each token's best occurrence
//          (longest extended match, then nearest) costs log2(distance), clamped to [0, log2 V]; match length only
//          selects the occurrence.
// BUFFER caps the suffix match length l. WINDOW is the whole lookback (buffer included), as in the reference code.
// The dist form also keeps the reference code's source range (positions at least BUFFER into the window).
// Only generated tokens enter the window (the reference code's history starts empty at the first output token).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace h5::lz {

enum class Form { Paper, Dist };
enum class Scope { Output, Think };   // H5_LZ_SCOPE: every generated token, or only while a <think> block is open

struct Config {
  float alpha;
  int window, buffer;
  Form form;
  Scope scope;
};

// Parses H5_LZ=ALPHA:WINDOW:BUFFER:FORM and H5_LZ_SCOPE. Returns "" on success, else the error.
inline std::string parse(const char* text, const char* scope, Config& c) {
  if (!text || !*text) return "H5_LZ is empty";
  char form[16] = {0}; float a = 0.f; int w = 0, b = 0; char tail = 0;
  if (std::sscanf(text, "%f:%d:%d:%15[a-z]%c", &a, &w, &b, form, &tail) != 4)
    return "invalid H5_LZ (ALPHA:WINDOW:BUFFER:paper|dist)";
  if (!(a >= 0.f) || !(a <= 4.f)) return "H5_LZ ALPHA must be in [0, 4] (0: off unless a request sets lz_penalty)";
  if (w < 16 || w > 4096) return "H5_LZ WINDOW must be 16..4096";
  if (b < 1 || b >= w) return "H5_LZ BUFFER must be 1..WINDOW-1";
  Form f;
  if (!std::strcmp(form, "paper")) f = Form::Paper; else if (!std::strcmp(form, "dist")) f = Form::Dist;
  else return "H5_LZ FORM must be paper or dist";
  Scope s;
  if (!scope) return "H5_LZ needs H5_LZ_SCOPE=output|think";
  if (!std::strcmp(scope, "output")) s = Scope::Output; else if (!std::strcmp(scope, "think")) s = Scope::Think;
  else return "H5_LZ_SCOPE must be output or think";
  c = Config{a, w, b, f, s};
  return "";
}

inline const char* name(Form f) { return f == Form::Paper ? "paper" : "dist"; }

// Per-thread scratch: an epoch-stamped token table over the vocabulary, so a row costs O(WINDOW + matches * BUFFER)
// with no clearing.
class Scratch {
 public:
  explicit Scratch(int vocab) : stamp_(vocab, 0), slot_(vocab, 0), log2v_(std::log2((double)vocab)) {}
  double log2v() const { return log2v_; }
  int vocab() const { return (int)stamp_.size(); }
  void begin() { if (++epoch_ == 0) { std::fill(stamp_.begin(), stamp_.end(), 0u); epoch_ = 1; } toks_.clear(); }
  // Returns the entry index of token t, creating it with the given initial state if new.
  struct Entry { int tok; int best_x; long best_d; long near_d; };
  Entry& at(int t, bool& fresh) {
    fresh = stamp_[t] != epoch_;
    if (fresh) { stamp_[t] = epoch_; slot_[t] = (uint32_t)toks_.size(); toks_.push_back(Entry{t, 0, 0, 0}); }
    return toks_[slot_[t]];
  }
  std::vector<Entry>& entries() { return toks_; }
 private:
  std::vector<uint32_t> stamp_, slot_;
  uint32_t epoch_ = 0;
  double log2v_;
  std::vector<Entry> toks_;
};

// Penalties for the row whose context is gen[0..ngen) followed by ext[0..k) (gen: generated tokens only).
// Calls emit(token, delta) once per token present in the window, delta <= 0; at most WINDOW calls.
template <class Emit>
inline void row(const Config& c, const int* gen, long ngen, const int* ext, int k, Scratch& s, Emit&& emit) {
  const long T = ngen + k;   // the candidate sits at index T
  if (T < 1) return;
  auto at = [&](long i) { return i < ngen ? gen[i] : ext[i - ngen]; };
  const long lo = std::max(0L, T - c.window);
  s.begin();
  int lmax = 0;   // longest suffix match over the window (capped at BUFFER)
  // Source position p: candidate at(p) copied from p, distance T - p; its match length is 1 + the common suffix
  // length of [lo, p) and [lo, T) (capped at BUFFER). Scan nearest first so the first sighting is the nearest.
  // The dist form follows the reference code exactly: sources start BUFFER tokens into the window (each needs a
  // full BUFFER-long preceding slice) and a window of at most BUFFER tokens penalises nothing.
  const long src_lo = c.form == Form::Dist ? lo + c.buffer : lo;
  for (long p = T - 1; p >= src_lo; --p) {
    int u = 0;
    while (u < c.buffer && p - 1 - u >= lo && at(p - 1 - u) == at(T - 1 - u)) ++u;
    const int x = u + 1;
    const long d = T - p;
    bool fresh;
    auto& e = s.at(at(p), fresh);
    if (fresh) { e.near_d = d; e.best_x = x; e.best_d = d; }
    else if (x > e.best_x) { e.best_x = x; e.best_d = d; }   // strictly longer; ties keep the nearer (earlier seen)
    lmax = std::max(lmax, u);
  }
  const double lv = s.log2v();
  for (const auto& e : s.entries()) {
    double bits;
    if (c.form == Form::Dist) bits = std::min(lv, std::max(0.0, std::log2((double)e.best_d)));
    else if (lmax >= 1 && e.best_x == lmax + 1) {
      const double l = lmax, d = std::max((double)e.best_d, l + 2.0);
      bits = std::log2(1.0 + (d - l - 1.0) / (l * d)) - 1.0;
    } else bits = std::log2((double)e.near_d);
    const float delta = (float)(-c.alpha * (lv - bits));
    if (delta < 0.f) emit(e.tok, delta);
  }
}

}  // namespace h5::lz

namespace h5::adj {
// Per-row accumulator for host-built logit adjustments (DRY, LZ, fixed token biases): sums entries per token so
// each token appears once in the row (k_logit_adj requires unique tokens), then writes the row's K slots.
class Row {
 public:
  explicit Row(int vocab) : stamp_(vocab, 0), slot_(vocab, 0) {}
  void begin() { if (++epoch_ == 0) { std::fill(stamp_.begin(), stamp_.end(), 0u); epoch_ = 1; } tok_.clear(); val_.clear(); }
  void add(int t, float v) {
    if (stamp_[t] != epoch_) { stamp_[t] = epoch_; slot_[t] = (uint32_t)tok_.size(); tok_.push_back(t); val_.push_back(v); }
    else val_[slot_[t]] += v;
  }
  size_t size() const { return tok_.size(); }
  // Writes the row into K slots (unused: token -1). Returns false if the row holds more than K tokens.
  bool write(int K, int* tok, float* val) const {
    if (tok_.size() > (size_t)K) return false;
    for (int j = 0; j < K; ++j) { tok[j] = j < (int)tok_.size() ? tok_[j] : -1; val[j] = j < (int)tok_.size() ? val_[j] : 0.f; }
    return true;
  }
 private:
  std::vector<uint32_t> stamp_, slot_;
  uint32_t epoch_ = 0;
  std::vector<int> tok_;
  std::vector<float> val_;
};
}  // namespace h5::adj
