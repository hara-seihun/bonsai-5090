#pragma once
// Loop-safe relaxed acceptance (H5_RELAX_POLICY). See tools/LOOPSAFE.md.
//
// Under H5_DRAFT_MARGIN the verifier accepts a proposal whose logit is within the margin of the argmax. In long
// reasoning that substitution sustains repetition loops: wherever greedy would leave a repeated passage (its
// argmax departs from the earlier text) the copy is usually the runner-up, and the drafter and prompt lookup
// propose exactly the copy. A policy here vetoes individual margin substitutions on the host: the engine reports
// each row's exact greedy token (H5_RELAX_STATS), and a vetoed row reverts to greedy, so the accepted prefix stops
// there and the round's bonus is the target's own argmax. Rows the target already agrees with are never touched,
// so every policy lies between strict (veto all) and the plain margin (veto none).
//
// Grammar: comma-separated rules, all of which must allow a substitution:
//   ngram=N[:W]   veto if the proposal completes an N-token n-gram already present in the last W tokens of the
//                 sequence (prompt + output + this round's earlier proposals); W 0 or omitted = whole sequence
//   p1=T          allow only if the target's top-1 probability is below T              (needs stats 2)
//   ent=H         allow only if the target's entropy exceeds H nats                       (needs stats 2)
//   p1min=T       allow only if the target's top-1 probability is at least T (a confident two-way split) (stats 2)
//   entmax=H      allow only if the target's entropy is below H nats                         (needs stats 2)
//   firstk=K      allow only at proposal positions 0..K-1 of a round
//   lookup        veto proposals that came from the prompt-lookup extension
//   every=N       every N-th verify round of a request is strict
//   decay=L       margin shrinks linearly with output length: margin * max(0, 1 - generated / L)
//   guard=N:M     when the last N output tokens already occurred earlier in the sequence, the next M rounds
//                 are strict (a loop detector that breaks a cycle once it is underway)
//   maxsub=K      at most K substitutions per round (later ones veto)
//   digit         veto when the proposal or the target's argmax contains a digit
//   special       veto when the proposal or the target's argmax is a control token (e.g. <think>, </think>)
//   escape=L:M    (changes the target's own choice, not just acceptance) when the target's argmax would extend a
//                 verbatim repeat of at least L tokens (the L-gram ending in it occurred earlier) and its runner-up
//                 is within M logits, emit the runner-up instead and end the round there. A minimal repetition
//                 penalty that acts only inside long verbatim copies; it can break loops greedy itself falls into.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include "draft_contract.h"

namespace h5::relax {

struct Policy {
  bool active = false;
  int ngram = 0, ngram_window = 0;
  float p1 = 0.f, ent = 0.f, p1min = 0.f, entmax = 0.f;
  int firstk = 0;
  bool lookup = false;
  int every = 0;
  float decay = 0.f;
  int guard_n = 0, guard_m = 0;
  int maxsub = -1;
  bool digit = false, special = false;
  int escape_n = 0; float escape_m = 0.f;
  int stats_mode() const { return !active ? 0 : (p1 > 0.f || ent > 0.f || p1min > 0.f || entmax > 0.f) ? 2 : 1; }
  bool needs_bonus_row() const { return escape_n > 0; }
};

enum class Reason { Allowed, Ngram, P1, Entropy, FirstK, Lookup, Every, Decay, Guard, MaxSub, Digit, Escape, Special, Count };
inline const char* name(Reason r) {
  static const char* n[] = {"allowed", "ngram", "p1", "entropy", "firstk", "lookup", "every", "decay", "guard", "maxsub", "digit", "escape", "special"};
  return n[(int)r];
}

inline bool parse_int(const std::string& s, int& out, int lo) {
  char* e = nullptr; long v = std::strtol(s.c_str(), &e, 10);
  if (e == s.c_str() || *e || v < lo || v > 1 << 24) return false;
  out = (int)v; return true;
}
inline bool parse_float(const std::string& s, float& out) {
  char* e = nullptr; out = std::strtof(s.c_str(), &e);
  return e != s.c_str() && !*e && std::isfinite(out) && out > 0.f;
}
// Returns an empty string on success, else the error.
inline std::string parse(const char* text, Policy& p) {
  p = Policy{};
  if (!text || !*text || !std::strcmp(text, "none")) return "";
  std::string all(text); size_t at = 0;
  while (at <= all.size()) {
    size_t comma = all.find(',', at); if (comma == std::string::npos) comma = all.size();
    std::string item = all.substr(at, comma - at); at = comma + 1;
    std::string key = item, val; size_t eq = item.find('=');
    if (eq != std::string::npos) { key = item.substr(0, eq); val = item.substr(eq + 1); }
    auto two = [&](int& a, int& b, int alo, int blo, bool b_required) {
      size_t c = val.find(':');
      if (c == std::string::npos) { b = 0; return !b_required && parse_int(val, a, alo); }
      return parse_int(val.substr(0, c), a, alo) && parse_int(val.substr(c + 1), b, blo);
    };
    bool ok;
    if (key == "ngram") ok = two(p.ngram, p.ngram_window, 1, 0, false);
    else if (key == "p1") ok = parse_float(val, p.p1) && p.p1 <= 1.f;
    else if (key == "ent") ok = parse_float(val, p.ent);
    else if (key == "p1min") ok = parse_float(val, p.p1min) && p.p1min <= 1.f;
    else if (key == "entmax") ok = parse_float(val, p.entmax);
    else if (key == "firstk") ok = parse_int(val, p.firstk, 1);
    else if (key == "lookup") ok = val.empty() && (p.lookup = true);
    else if (key == "every") ok = parse_int(val, p.every, 1);
    else if (key == "decay") ok = parse_float(val, p.decay);
    else if (key == "guard") ok = two(p.guard_n, p.guard_m, 1, 1, true);
    else if (key == "maxsub") ok = parse_int(val, p.maxsub, 0);
    else if (key == "digit") ok = val.empty() && (p.digit = true);
    else if (key == "special") ok = val.empty() && (p.special = true);
    else if (key == "escape") {
      size_t c = val.find(':');
      ok = c != std::string::npos && parse_int(val.substr(0, c), p.escape_n, 2) && parse_float(val.substr(c + 1), p.escape_m);
    }
    else return "unknown H5_RELAX_POLICY rule: " + key;
    if (!ok) return "invalid H5_RELAX_POLICY value: " + item;
    if (comma == all.size()) break;
  }
  p.active = true;
  return "";
}

// Per-request sequence (prompt + emitted tokens) with exact n-gram indexes for a few lengths.
class History {
 public:
  void configure(int n, int guard_n, int escape_n = 0) {
    guard_n_ = guard_n;
    for (int m : {n, escape_n}) if (m > 0 && slot(m) < 0) idx_.push_back({m, {}});
  }
  void append(const int* t, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      seq_.push_back(t[i]);
      for (auto& [m, map] : idx_) if (seq_.size() >= (size_t)m) map[hash(seq_.size() - m, m)] = seq_.size();
      if (guard_n_ && seq_.size() >= (size_t)guard_n_) {
        // Guard: is the suffix ending here a repeat? Look it up before indexing this end.
        const uint64_t h = hash(seq_.size() - guard_n_, guard_n_);
        auto it = guard_.find(h);
        guard_hit_ = it != guard_.end() && same(it->second - guard_n_, seq_.size() - guard_n_, guard_n_);
        guard_[h] = seq_.size();
      }
    }
  }
  bool guard_hit() const { return guard_hit_; }
  size_t size() const { return seq_.size(); }
  // Does the n-gram tail(n-1) + candidate, where tail is the sequence followed by ext[0..k-1], occur earlier within
  // the last `window` tokens (0 = anywhere)? n must have been configured.
  bool repeats(int n, const int* ext, int k, int candidate, int window) const {
    const int si = slot(n);
    if (si < 0) return false;
    std::vector<int> g(n);
    g[n - 1] = candidate;
    for (int j = n - 2, src = k - 1; j >= 0; --j, --src) {
      if (src >= 0) g[j] = ext[src];
      else { const long q = (long)seq_.size() + src; if (q < 0) return false; g[j] = seq_[q]; }
    }
    const size_t total = seq_.size() + k;   // candidate would sit at index `total`
    const auto& map = idx_[si].second;
    auto it = map.find(hash_of(g.data(), n));
    if (it != map.end()) {
      const size_t e = it->second;
      bool eq = true;
      for (int j = 0; j < n && eq; ++j) eq = seq_[e - n + j] == g[j];
      if (eq && (!window || total + 1 - e <= (size_t)window)) return true;
    }
    // Occurrences ending in the round's proposals (not indexed): brute force over the short tail.
    for (int end = 1; end <= k; ++end) {   // n-gram ending (exclusive) at seq_.size() + end
      const long e = (long)seq_.size() + end;
      if (e < n) continue;
      bool eq = true;
      for (int j = 0; j < n && eq; ++j) {
        const long ix = e - n + j;
        eq = (ix < (long)seq_.size() ? seq_[ix] : ext[ix - seq_.size()]) == g[j];
      }
      if (eq) return true;
    }
    return false;
  }

 private:
  int slot(int n) const { for (size_t i = 0; i < idx_.size(); ++i) if (idx_[i].first == n) return (int)i; return -1; }
  static uint64_t mix(uint64_t h, int t) { h ^= (uint32_t)t; h *= UINT64_C(1099511628211); return h ^ (h >> 29); }
  static uint64_t hash_of(const int* t, int n) { uint64_t h = UINT64_C(1469598103934665603) ^ (uint64_t)n; for (int k = 0; k < n; ++k) h = mix(h, t[k]); return h; }
  uint64_t hash(size_t b, int n) const { return hash_of(seq_.data() + b, n); }
  bool same(size_t a, size_t b, int n) const { for (int k = 0; k < n; ++k) if (seq_[a + k] != seq_[b + k]) return false; return true; }
  int guard_n_ = 0;
  bool guard_hit_ = false;
  std::vector<int> seq_;
  std::vector<std::pair<int, std::unordered_map<uint64_t, size_t>>> idx_;   // most recent end per n-gram
  std::unordered_map<uint64_t, size_t> guard_;
};

// Per-request round state.
struct State {
  History history;
  long rounds = 0;
  int guard_left = 0;
  int generated = 0;
};

struct Counters { long offered = 0, by_reason[(int)Reason::Count] = {}; };

// Token class table: bit 0 = contains a digit, bit 1 = control token.
inline int is_digit_token(int t, const std::vector<char>& digit_table) { return t >= 0 && t < (int)digit_table.size() ? digit_table[t] : 0; }

// Apply the policy to one verify round. out[k] is the engine's (relaxed) output at row k (k = 0..nd; row nd is the
// bonus, read only by the escape rule),
// st[k] its row statistics, dr the proposals, lookup_from the first proposal index that came from prompt lookup
// (nd if none). Vetoed substitutions are replaced by the exact greedy token. Rows after the first mismatch are
// irrelevant to the accepted prefix and are left as they are.
inline void apply(const Policy& p, State& s, int* out, const RowStat* st, const int* dr, int nd, int lookup_from,
                  float margin, const std::vector<char>& digit_table, Counters& c) {
  if (!p.active) return;
  const bool guard_strict = s.guard_left > 0;
  const bool every_strict = p.every && s.rounds % p.every == p.every - 1;
  s.rounds++;
  if (s.guard_left > 0) s.guard_left--;
  int subs = 0;
  const int last = p.escape_n ? nd : nd - 1;   // row nd (the bonus) is read only by the escape rule
  for (int k = 0; k <= last; ++k) {
    if (k < nd && out[k] == dr[k] && st[k].greedy >= 0 && st[k].greedy != dr[k]) {
      // margin substitution: the policy may veto it
      c.offered++;
      Reason r = Reason::Allowed;
      if (guard_strict) r = Reason::Guard;
      else if (every_strict) r = Reason::Every;
      else if (p.firstk && k >= p.firstk) r = Reason::FirstK;
      else if (p.lookup && k >= lookup_from) r = Reason::Lookup;
      else if (p.maxsub >= 0 && subs >= p.maxsub) r = Reason::MaxSub;
      else if (p.digit && ((is_digit_token(dr[k], digit_table) & 1) || (is_digit_token(st[k].greedy, digit_table) & 1))) r = Reason::Digit;
      else if (p.special && ((is_digit_token(dr[k], digit_table) & 2) || (is_digit_token(st[k].greedy, digit_table) & 2))) r = Reason::Special;
      else if (p.decay > 0.f && st[k].best - st[k].score > margin * std::fmax(0.f, 1.f - (s.generated + k) / p.decay)) r = Reason::Decay;
      else if (p.p1 > 0.f && !(std::exp(st[k].best - st[k].lse) < p.p1)) r = Reason::P1;
      else if (p.ent > 0.f && !(st[k].entropy > p.ent)) r = Reason::Entropy;
      else if (p.p1min > 0.f && !(std::exp(st[k].best - st[k].lse) >= p.p1min)) r = Reason::P1;
      else if (p.entmax > 0.f && !(st[k].entropy < p.entmax)) r = Reason::Entropy;
      else if (p.ngram && s.history.repeats(p.ngram, dr, k, dr[k], p.ngram_window)) r = Reason::Ngram;
      c.by_reason[(int)r]++;
      if (r == Reason::Allowed) { subs++; continue; }
      out[k] = st[k].greedy;
    }
    // Here out[k] is the target's greedy token (an exact match, a vetoed substitution, a mismatch or the bonus).
    if (p.escape_n && st[k].greedy >= 0 && out[k] == st[k].greedy && st[k].second >= 0 && !draft_eos(st[k].second) &&
        !draft_eos(st[k].greedy) && st[k].best - st[k].second_logit <= p.escape_m &&
        s.history.repeats(p.escape_n, dr, k, st[k].greedy, 0)) {
      out[k] = st[k].second;
      c.by_reason[(int)Reason::Escape]++;   // the runner-up may equal the proposal: then the prefix continues
    }
    if (k == last || out[k] != dr[k]) break;   // the accepted prefix ends here
  }
}
// At admission: the prompt joins the history without arming the guard.
inline void seed(const Policy& p, State& s, const int* prompt, int count) {
  s = State{};
  if (!p.active) return;
  s.history.configure(p.ngram, p.guard_n, p.escape_n);
  s.history.append(prompt, count);
}
// After the round's tokens are emitted: extend the history and arm the guard.
inline void observe(const Policy& p, State& s, const int* emitted, int count) {
  if (!p.active) return;
  for (int i = 0; i < count; ++i) {
    s.history.append(emitted + i, 1);
    if (p.guard_n && s.history.guard_hit() && s.guard_left == 0) s.guard_left = p.guard_m;
  }
  s.generated += count;
}

}  // namespace h5::relax

namespace h5::dry {
// H5_DRY=MULT:BASE:LEN[:WINDOW] — DRY ("don't repeat yourself") penalty on the target's logits, applied to every
// verify row with that row's own context (committed sequence + the round's earlier proposals). A token that would
// extend a verbatim repeat whose current matched suffix has length n >= LEN loses MULT * BASE^(n - LEN) logits
// (capped at CAP). Unlike the relax policy this changes the target's choice (greedy included), so it can break
// loops strict greedy falls into. Needs H5_LOGIT_ADJ >= K adjustments per row.
enum class Scope { All, Output, Think };   // H5_DRY_SCOPE: match over prompt+output, output only, or output only
                                           // and only while a <think> block is open
struct Config { bool active = false; float mult = 0.f, base = 0.f; int len = 0, window = 4096; Scope scope = Scope::All; };
inline bool parse_scope(const char* t, Scope& s) {
  if (!t) return false;
  if (!std::strcmp(t, "all")) s = Scope::All; else if (!std::strcmp(t, "output")) s = Scope::Output;
  else if (!std::strcmp(t, "think")) s = Scope::Think; else return false;
  return true;
}
constexpr float CAP = 30.f;
constexpr int MAX_MATCH = 256;
inline std::string parse(const char* text, Config& c) {
  c = Config{};
  if (!text || !*text || !std::strcmp(text, "none")) return "";
  float v[4] = {0, 0, 0, 4096}; int n = 0; const char* p = text;
  while (*p && n < 4) { char* e; v[n++] = std::strtof(p, &e); if (e == p) return "invalid H5_DRY"; p = *e == ':' ? e + 1 : e; if (*e && *e != ':') return "invalid H5_DRY"; }
  if (n < 3 || *p || !(v[0] > 0.f) || !(v[1] >= 1.f) || v[2] < 1.f || v[3] < 16.f || v[3] > (1 << 22)) return "invalid H5_DRY (MULT>0:BASE>=1:LEN>=1[:WINDOW>=16])";
  c = Config{true, v[0], v[1], (int)v[2], (int)v[3]};
  return "";
}
// Penalties for the row whose context is seq followed by ext[0..k-1]. Writes up to K (token, -penalty) pairs into
// tok/val (rest token -1); keeps the K largest.
// start: first sequence index that may take part in a match (H5_DRY_SCOPE=output|think: the prompt length).
inline void row(const Config& c, const std::vector<int>& seq, const int* ext, int k, int K, int* tok, float* val, long start = 0) {
  for (int j = 0; j < K; ++j) { tok[j] = -1; val[j] = 0.f; }
  const long total = (long)seq.size() + k;
  if (!c.active || total < 2) return;
  auto at = [&](long i) { return i < (long)seq.size() ? seq[i] : ext[i - seq.size()]; };
  const int last = at(total - 1);
  const long lo = std::max(start, total - 1 - c.window);
  int filled = 0;
  for (long i = total - 2; i >= lo; --i) {   // earlier occurrence of the last token at i; candidate at(i + 1)
    if (at(i) != last) continue;
    int n = 1;
    while (n < MAX_MATCH && i - n >= start && at(i - n) == at(total - 1 - n)) ++n;
    if (n < c.len) continue;
    const int cand = at(i + 1);
    const float pen = std::fmin(CAP, c.mult * std::pow(c.base, (float)(n - c.len)));
    int slot = -1;
    for (int j = 0; j < filled; ++j) if (tok[j] == cand) { slot = j; break; }
    if (slot >= 0) { if (-pen < val[slot]) val[slot] = -pen; continue; }
    if (filled < K) { tok[filled] = cand; val[filled] = -pen; ++filled; continue; }
    int weakest = 0; for (int j = 1; j < K; ++j) if (val[j] > val[weakest]) weakest = j;
    if (-pen < val[weakest]) { tok[weakest] = cand; val[weakest] = -pen; }
  }
}
}  // namespace h5::dry

namespace h5::loopclose {
// H5_LOOP_CLOSE=MAXP:MINSPAN — loop breaker for reasoning (vLLM #52677 / MLX-fork style). While a <think> block is
// open, if the output's last max(3p, MINSPAN) tokens are exactly periodic with some period p <= MAXP, the round's
// bonus token is replaced by </think>, so the model stops reasoning and answers from what it has.
struct Config { bool active = false; int maxp = 0, min_span = 0; };
inline std::string parse(const char* text, Config& c) {
  c = Config{};
  if (!text || !*text || !std::strcmp(text, "none")) return "";
  int a = 0, b = 0; char tail = 0;
  if (std::sscanf(text, "%d:%d%c", &a, &b, &tail) != 2 || a < 1 || a > 4096 || b < 2 || b > 65536) return "invalid H5_LOOP_CLOSE (MAXP:MINSPAN)";
  c = Config{true, a, b};
  return "";
}
// Smallest period p <= maxp such that the last max(3p, min_span) tokens of t (followed by ext[0..k-1]) repeat with
// period p; 0 if none.
inline int period(const Config& c, const std::vector<int>& t, const int* ext, int k) {
  const long n = (long)t.size() + k;
  auto at = [&](long i) { return i < (long)t.size() ? t[i] : ext[i - t.size()]; };
  for (int p = 1; p <= c.maxp; ++p) {
    const long span = std::max<long>(3L * p, c.min_span);
    if (span > n) break;
    bool ok = true;
    for (long i = n - 1; i >= n - span + p && ok; --i) ok = at(i) == at(i - p);
    if (ok) return p;
  }
  return 0;
}
}  // namespace h5::loopclose
