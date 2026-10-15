// CPU tests for src/lz_penalty.h (H5_LZ) and the per-row adjustment accumulator. `make lz-penalty-test`.
#include "lz_penalty.h"
#include "lz_penalty_vectors.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

using Pen = std::map<int, float>;
static Pen run(const h5::lz::Config& c, const std::vector<int>& gen, const std::vector<int>& ext, int k, h5::lz::Scratch& s) {
  Pen p; int calls = 0;
  h5::lz::row(c, gen.data(), (long)gen.size(), ext.data(), k, s, [&](int t, float v) { CHECK(!p.count(t)); CHECK(v < 0.f); p[t] = v; ++calls; });
  CHECK(calls <= c.window);
  return p;
}

// Paper form straight from the definitions, O(W^2 B): longest suffix match l over the window, extension set =
// tokens following an occurrence of an l-long match, everything else in the window costs log2(nearest distance).
static Pen paper_brute(const h5::lz::Config& c, const std::vector<int>& seq, int vocab) {
  Pen p; const long T = seq.size(); if (T < 1) return p;
  const long lo = std::max(0L, T - c.window);
  auto suffix = [&](long p0) { int u = 0; while (u < c.buffer && p0 - 1 - u >= lo && seq[p0 - 1 - u] == seq[T - 1 - u]) ++u; return u; };
  int l = 0; for (long q = lo; q < T; ++q) l = std::max(l, suffix(q));
  const double lv = std::log2((double)vocab);
  for (int t = 0; t < vocab; ++t) {
    long near = -1, ext = -1;
    for (long q = T - 1; q >= lo; --q) if (seq[q] == t) {
      if (near < 0) near = T - q;
      if (l >= 1 && suffix(q) == l && ext < 0) ext = T - q;
    }
    if (near < 0) continue;
    double bits;
    if (ext > 0) { const double d = std::max((double)ext, l + 2.0); bits = std::log2(1.0 + (d - l - 1.0) / (l * d)) - 1.0; }
    else bits = std::log2((double)near);
    const float v = (float)(-c.alpha * (lv - bits));
    if (v < 0.f) p[t] = v;
  }
  return p;
}

static bool close(float a, float b) { return std::fabs(a - b) <= 1e-4f * std::max(1.f, std::fabs(b)); }

int main() {
  // --- parsing
  h5::lz::Config c{};
  CHECK(h5::lz::parse("0.15:512:32:paper", "think", c).empty() && c.window == 512 && c.buffer == 32 && c.form == h5::lz::Form::Paper && c.scope == h5::lz::Scope::Think);
  CHECK(h5::lz::parse("0.1:256:16:dist", "output", c).empty() && c.form == h5::lz::Form::Dist && c.scope == h5::lz::Scope::Output);
  CHECK(!h5::lz::parse("0.15:512:32:paper", nullptr, c).empty());          // scope required
  CHECK(!h5::lz::parse("0.15:512:32", "output", c).empty());               // form required
  CHECK(!h5::lz::parse("0.15:512:32:gzip", "output", c).empty());
  CHECK(h5::lz::parse("0:512:32:paper", "output", c).empty() && c.alpha == 0.f);   // shape only; requests opt in
  CHECK(!h5::lz::parse("-0.1:512:32:paper", "output", c).empty());
  CHECK(!h5::lz::parse("4.5:512:32:paper", "output", c).empty());
  CHECK(!h5::lz::parse("0.15:512:512:paper", "output", c).empty());        // buffer < window
  CHECK(!h5::lz::parse("0.15:8:4:paper", "output", c).empty());
  CHECK(!h5::lz::parse("0.15:512:32:paperx", "output", c).empty());
  CHECK(!h5::lz::parse("0.15:512:32:paper", "all", c).empty());

  // --- dist form equals the authors' reference penalizer (NumPy port, bench/lz_penalty_vectors.h)
  {
    h5::lz::Scratch s(lzref::VOCAB);
    int checked = 0;
    for (const auto& rc : lzref::cases) {
      const float alpha = 0.15f;
      h5::lz::Config d{alpha, rc.window, rc.buffer, h5::lz::Form::Dist, h5::lz::Scope::Output};
      std::vector<int> gen(rc.seq, rc.seq + rc.n), none;
      const Pen p = run(d, gen, none, 0, s);
      for (int t = 0; t < lzref::VOCAB; ++t) {
        const float want = (float)(-alpha * rc.pen[t]);
        const float got = p.count(t) ? p.at(t) : 0.f;
        if (!close(got, want)) { std::printf("  case w=%d b=%d n=%d token %d: got %g want %g\n", rc.window, rc.buffer, rc.n, t, got, want); ++failures; }
        ++checked;
      }
    }
    std::printf("dist vs reference: %d token penalties\n", checked);
  }

  // --- paper form equals the brute-force definition; rows with proposals equal the extended sequence
  {
    std::mt19937 rng(7);
    const int V = 9;
    h5::lz::Scratch s(V);
    for (int it = 0; it < 400; ++it) {
      h5::lz::Config p{0.15f, 16 + (int)(rng() % 40), 1 + (int)(rng() % 8), it % 2 ? h5::lz::Form::Paper : h5::lz::Form::Dist, h5::lz::Scope::Output};
      if (p.buffer >= p.window) p.buffer = p.window - 1;
      const int n = (int)(rng() % 90);
      std::vector<int> seq(n);
      const int period = 1 + (int)(rng() % 7);
      for (int i = 0; i < n; ++i) seq[i] = (i > n / 3 && it % 3 == 0) ? seq[i - period >= 0 ? i - period : 0] : (int)(rng() % (it % 4 == 1 ? 3 : V));
      if (p.form == h5::lz::Form::Paper) {
        const Pen got = run(p, seq, {}, 0, s), want = paper_brute(p, seq, V);
        CHECK(got.size() == want.size());
        for (auto& [t, v] : want) CHECK(got.count(t) && close(got.at(t), v));
      }
      // Causal rows: row k over (gen, ext[0..k)) equals row 0 over gen ++ ext[0..k).
      const int k = std::min(n, (int)(rng() % 9));
      std::vector<int> gen(seq.begin(), seq.end() - k), ext(seq.end() - k, seq.end());
      ext.push_back(-7);   // a later proposal must never be read
      const Pen a = run(p, gen, ext, k, s), b = run(p, seq, {}, 0, s);
      CHECK(a == b);
    }
  }

  // --- mechanism: inside a loop the copy token pays the most; absent tokens and tokens beyond the window pay 0
  {
    const int V = 1000;
    h5::lz::Scratch s(V);
    h5::lz::Config p{0.15f, 64, 8, h5::lz::Form::Paper, h5::lz::Scope::Output};
    std::vector<int> seq;
    for (int i = 0; i < 100; ++i) seq.push_back(500 + i);          // novel text, scrolls out of the window
    for (int r = 0; r < 3; ++r) for (int i = 0; i < 20; ++i) seq.push_back(i);   // period-20 loop
    const Pen pen = run(p, seq, {}, 0, s);
    const int copy = 0;   // the loop continues with token 0
    float worst = 0.f; int worst_t = -1;
    for (auto& [t, v] : pen) if (v < worst) { worst = v; worst_t = t; }
    CHECK(worst_t == copy);
    CHECK(close(pen.at(copy), (float)(-0.15 * (std::log2(1000.0) - (std::log2(1.0 + (20.0 - 8 - 1) / (8.0 * 20)) - 1.0)))));
    CHECK(!pen.count(999));              // never seen
    CHECK(!pen.count(500));              // seen, but 120 tokens back with a 64-token window
    CHECK(pen.count(19) && pen.at(19) > pen.at(copy));   // last token: distance 1, not an extension (19 is followed by 0)
    // The dist form can rank the loop's copy below a nearer in-window token: the reference code's weakness.
    h5::lz::Config d = p; d.form = h5::lz::Form::Dist;
    const Pen dp = run(d, seq, {}, 0, s);
    CHECK(dp.count(copy) && dp.count(19) && dp.at(19) < dp.at(copy));
    // Overlapping (period 1) run: extension still gets the maximum penalty, finite.
    std::vector<int> run1(40, 7);
    const Pen r1 = run(p, run1, {}, 0, s);
    CHECK(r1.size() == 1 && std::isfinite(r1.at(7)) && r1.at(7) < (float)(-0.15 * std::log2(1000.0)));
  }

  // --- adjustment rows: per-token sums, unique tokens, capacity enforced
  {
    h5::adj::Row row(16);
    row.begin(); row.add(3, -1.f); row.add(5, -2.f); row.add(3, -0.5f);
    int tok[4]; float val[4];
    CHECK(row.size() == 2 && row.write(4, tok, val));
    CHECK(tok[0] == 3 && val[0] == -1.5f && tok[1] == 5 && val[1] == -2.f && tok[2] == -1 && tok[3] == -1 && val[2] == 0.f);
    CHECK(!row.write(1, tok, val));
    row.begin(); CHECK(row.size() == 0); row.add(3, -4.f); CHECK(row.write(4, tok, val) && val[0] == -4.f && tok[1] == -1);
  }

  if (failures) { std::printf("lz-penalty-test: %d failures\n", failures); return 1; }
  std::printf("lz-penalty-test: ok\n");
  return 0;
}
