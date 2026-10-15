// make relax-policy-test: host checks of relax_policy.h (parser, n-gram veto, guard, lookup/firstk/every rules).
#include "../src/relax_policy.h"
#include <cassert>
#include <cstdio>
using namespace h5;
using namespace h5::relax;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)
static RowStat sub(int greedy, float gap = 1.f) { return RowStat{greedy, 10.f, 10.f - gap, 11.f, 2.f, -1, 0.f}; }
int main() {
  Policy p;
  CHECK(parse("none", p).empty() && !p.active);
  CHECK(parse("ngram=4:2048,lookup,firstk=3", p).empty() && p.active && p.ngram == 4 && p.ngram_window == 2048 && p.lookup && p.firstk == 3);
  CHECK(!parse("ngram=0", p).empty());
  CHECK(!parse("bogus=1", p).empty());
  CHECK(parse("guard=32:4", p).empty() && p.guard_n == 32 && p.guard_m == 4);
  CHECK(!parse("guard=32", p).empty());
  CHECK(parse("p1=0.5", p).empty() && p.stats_mode() == 2);
  CHECK(parse("ngram=8", p).empty() && p.stats_mode() == 1);
  std::vector<char> digits(100, 0); digits[9] = 1; digits[77] = 2;
  // n-gram veto: history 1 2 3 4 5; proposals 1 2 3 4 [5]: the 4-gram 2 3 4 5 repeats at proposal 4.
  {
    parse("ngram=4", p); State s; int prompt[] = {1, 2, 3, 4, 5, 6}; seed(p, s, prompt, 6);
    int dr[] = {1, 2, 3, 4, 5, 7}; int out[] = {1, 2, 3, 4, 5, 7};
    RowStat st[] = {sub(1), sub(2), sub(3), sub(90), sub(91), sub(92)};
    // rows 0-2 exact; row 3 substitution: n-gram (1 2 3 4) = earlier → veto.
    Counters c; apply(p, s, out, st, dr, 6, 6, 1.5f, digits, c);
    CHECK(out[3] == 90 && c.by_reason[(int)Reason::Ngram] == 1);
  }
  {
    parse("ngram=4", p); State s; int prompt[] = {1, 2, 3, 4, 5, 6}; seed(p, s, prompt, 6);
    int dr[] = {8, 9, 10, 11}; int out[] = {8, 9, 10, 11};
    RowStat st[] = {sub(50), sub(9), sub(10), sub(11)};
    Counters c; apply(p, s, out, st, dr, 4, 4, 1.5f, digits, c);   // novel: allowed
    CHECK(out[0] == 8 && c.by_reason[0] == 1);
  }
  // Repeat formed inside the round's own proposals: 7 7 7 7 7 with N=3.
  {
    parse("ngram=3", p); State s; int prompt[] = {1, 2}; seed(p, s, prompt, 2);
    int dr[] = {7, 7, 7, 7}; int out[] = {7, 7, 7, 7};
    RowStat st[] = {sub(7), sub(7), sub(7), sub(5)};
    Counters c; apply(p, s, out, st, dr, 4, 4, 1.5f, digits, c);   // row 3: (7 7 7) occurred at rows 0-2 → veto
    CHECK(out[3] == 5 && c.by_reason[(int)Reason::Ngram] == 1);
  }
  // Window: an occurrence further back than W does not veto.
  {
    parse("ngram=2:4", p); State s; int prompt[] = {1, 2, 9, 9, 9, 9, 9, 9, 1}; seed(p, s, prompt, 9);
    int dr[] = {2}; int out[] = {2}; RowStat st[] = {sub(3)};
    Counters c; apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c);
    CHECK(out[0] == 2);
    parse("ngram=2", p); seed(p, s, prompt, 9); out[0] = 2; apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c);
    CHECK(out[0] == 3);
  }
  // lookup, firstk, every, digit, maxsub, decay
  {
    parse("lookup", p); State s; int prompt[] = {1}; seed(p, s, prompt, 1);
    int dr[] = {4, 5, 6}; int out[] = {4, 5, 6}; RowStat st[] = {sub(40), sub(50), sub(60)};
    Counters c; apply(p, s, out, st, dr, 3, 2, 1.5f, digits, c);
    CHECK(out[0] == 4 && out[1] == 5 && out[2] == 60);
    parse("firstk=1", p); seed(p, s, prompt, 1); int o2[] = {4, 5, 6}; apply(p, s, o2, st, dr, 3, 3, 1.5f, digits, c);
    CHECK(o2[0] == 4 && o2[1] == 50);
    parse("every=2", p); seed(p, s, prompt, 1); int o3[] = {4, 5, 6}; apply(p, s, o3, st, dr, 3, 3, 1.5f, digits, c);
    CHECK(o3[0] == 4); int o4[] = {4, 5, 6}; apply(p, s, o4, st, dr, 3, 3, 1.5f, digits, c); CHECK(o4[0] == 40);
    parse("maxsub=2", p); seed(p, s, prompt, 1); int o5[] = {4, 5, 6}; apply(p, s, o5, st, dr, 3, 3, 1.5f, digits, c);
    CHECK(o5[1] == 5 && o5[2] == 60);
    parse("digit", p); seed(p, s, prompt, 1); int d9[] = {9}; int o6[] = {9}; RowStat s9[] = {sub(8)}; apply(p, s, o6, s9, d9, 1, 1, 1.5f, digits, c);
    CHECK(o6[0] == 8);
    parse("special", p); seed(p, s, prompt, 1); int d77[] = {77}; int o8[] = {77}; RowStat s77[] = {sub(8)}; apply(p, s, o8, s77, d77, 1, 1, 1.5f, digits, c);
    CHECK(o8[0] == 8);
    parse("digit", p); seed(p, s, prompt, 1); o8[0] = 77; apply(p, s, o8, s77, d77, 1, 1, 1.5f, digits, c); CHECK(o8[0] == 77);
    parse("decay=100", p); seed(p, s, prompt, 1); s.generated = 60; int o7[] = {4}; RowStat sd[] = {sub(40, 1.f)};
    apply(p, s, o7, sd, dr, 1, 1, 1.5f, digits, c); CHECK(o7[0] == 40);   // margin 1.5*0.4 = 0.6 < gap 1
  }
  // guard: after a repeated 4-token suffix, the next 2 rounds are strict.
  {
    parse("guard=4:2", p); State s; int prompt[] = {1, 2, 3, 4, 5}; seed(p, s, prompt, 5);
    int em[] = {1, 2, 3, 4}; observe(p, s, em, 4); CHECK(s.guard_left == 2);
    int dr[] = {6}; int out[] = {6}; RowStat st[] = {sub(60)}; Counters c;
    apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c); CHECK(out[0] == 60 && c.by_reason[(int)Reason::Guard] == 1);
  }
  // p1 / entropy (best 10, lse 11 → p1 = e^-1 ≈ 0.37; entropy 2)
  {
    parse("p1=0.3", p); State s; int prompt[] = {1}; seed(p, s, prompt, 1);
    int dr[] = {4}; int out[] = {4}; RowStat st[] = {sub(40)}; Counters c;
    apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c); CHECK(out[0] == 40);
    parse("p1=0.5", p); seed(p, s, prompt, 1); out[0] = 4; apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c); CHECK(out[0] == 4);
    parse("p1min=0.3", p); seed(p, s, prompt, 1); out[0] = 4; apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c); CHECK(out[0] == 4);
    parse("p1min=0.5", p); seed(p, s, prompt, 1); out[0] = 4; apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c); CHECK(out[0] == 40);
    parse("entmax=1.5", p); seed(p, s, prompt, 1); out[0] = 4; apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c); CHECK(out[0] == 40);
    parse("ent=2.5", p); seed(p, s, prompt, 1); out[0] = 4; apply(p, s, out, st, dr, 1, 1, 1.5f, digits, c); CHECK(out[0] == 40);
  }
  // escape: history 1..8 then 1..7; greedy continues the copy with 8, runner-up 50 within 1 logit → emitted 50.
  {
    parse("escape=4:1", p); State s; int prompt[] = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3}; seed(p, s, prompt, 11);
    auto e = [](int g, int sec, float gap) { return RowStat{g, 10.f, 10.f, 11.f, 2.f, sec, 10.f - gap}; };
    int dr[] = {4, 5, 6}; int out[] = {4, 5, 6, 9};
    RowStat st[] = {e(4, 40, 0.5f), e(5, 50, 2.f), e(6, 60, 2.f), e(9, 90, 2.f)};
    Counters c; apply(p, s, out, st, dr, 3, 3, 1.5f, digits, c);
    CHECK(out[0] == 40 && c.by_reason[(int)Reason::Escape] == 1);
    int out2[] = {4, 5, 6, 9}; RowStat st2[] = {e(4, 40, 2.f), e(5, 50, 2.f), e(6, 60, 0.5f), e(9, 90, 2.f)};
    apply(p, s, out2, st2, dr, 3, 3, 1.5f, digits, c);   // (2 3 4 5),(3 4 5 6) continue the copy; row 2 escapes
    CHECK(out2[0] == 4 && out2[1] == 5 && out2[2] == 60);
    int dr3[] = {9, 9, 9}; int out3[] = {9, 9, 9, 9}; RowStat st3[] = {e(9, 1, 0.5f), e(9, 1, 0.5f), e(9, 1, 0.5f), e(9, 1, 0.5f)};
    apply(p, s, out3, st3, dr3, 3, 3, 1.5f, digits, c);   // novel 4-grams (no earlier 9s): untouched
    CHECK(out3[0] == 9 && out3[1] == 9 && out3[2] == 9 && out3[3] == 9);
  }
  // DRY: seq 1 2 3 4 5 1 2 3 4 → candidate 5 extends a 4-token match: penalty 1 * 2^(4-2) = 4.
  {
    dry::Config d; CHECK(dry::parse("1:2:2", d).empty() && d.active && d.len == 2);
    CHECK(!dry::parse("1:0.5:2", d).empty());
    dry::parse("1:2:2", d);
    std::vector<int> seq = {1, 2, 3, 4, 5, 1, 2, 3}; int ext[] = {4}; int tok[4]; float val[4];
    dry::row(d, seq, ext, 1, 4, tok, val);
    CHECK(tok[0] == 5 && std::fabs(val[0] + 4.f) < 1e-5 && tok[1] == -1);
    dry::row(d, seq, ext, 0, 4, tok, val);   // suffix ...1 2 3: candidate 4 with n = 3 → 2
    CHECK(tok[0] == 4 && std::fabs(val[0] + 2.f) < 1e-5);
    std::vector<int> s2 = {7, 8, 9}; dry::row(d, s2, ext, 0, 4, tok, val); CHECK(tok[0] == -1);
  }
  // loop close: period detection over output + accepted drafts
  {
    loopclose::Config lc; CHECK(loopclose::parse("8:12", lc).empty() && lc.active);
    std::vector<int> t = {5, 1, 2, 3, 1, 2, 3, 1, 2}; int ext[] = {3, 1, 2, 3};
    CHECK(loopclose::period(lc, t, ext, 0) == 0);          // span 12 > 9
    CHECK(loopclose::period(lc, t, ext, 4) == 3);          // 13 tokens, last 12 periodic with p 3
    std::vector<int> u(20, 9); CHECK(loopclose::period(lc, u, nullptr, 0) == 1);
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}; CHECK(loopclose::period(lc, v, nullptr, 0) == 0);
  }
  printf(fails ? "relax-policy-test: %d FAILED\n" : "relax-policy-test: ok\n", fails);
  return fails != 0;
}
