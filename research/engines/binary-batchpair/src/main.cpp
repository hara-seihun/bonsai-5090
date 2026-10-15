// halo-5090 runner: reference check, greedy generation, batched benchmark.
//   h5 check MODEL REF.bin            teacher-forced logits (KLD, PPL, top-1, max error), prefill and decode paths
//   h5 decodebench MODEL REF.bin STEPS REPEATS  real teacher-forced single-stream decode
//   h5 gen MODEL PROMPT N [--chat]    greedy text
//   h5 bench MODEL STREAMS CTX STEPS  batched decode throughput (one token per stream per step)
#include "llama.h"
#include "head_demand.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <limits>
#include <new>

extern "C" {
void* h5_create(const char* path, int slots, int ctx, int rows);
void* h5_create_draft(const char* path, int slots, int ctx, int rows, int proposals);
void h5_pretune(void* e);
int h5_step(void* e, int R, const int* slot, const int* tok, const int* pos, int* out);
int h5_step_logits(void* e, int R, const int* slot, const int* tok, const int* pos, int* out, float* logits);
void h5_reset(void* e, int slot);
double h5_last_step_s(void* e);
void h5_gemm_bench(void* e, int R, int reps);
void h5_prof_report(void* e);
void h5_keep(void* e, int slot, int flag);
void h5_accept(void* e, int slot, int n);
int h5_draft_load(void* e, const char* path);
void h5_capture(void* e, int on);
void h5_copy_slot(void* e, int src, int dst, int npos);
const float* h5_draft_conf(void* e);
int h5_draft(void* e, int ns, const int* slot, const int* n_ing, const int* ing_row0, const int* ing_pos0, const int* anchor, const int* start, int* drafts);
}
static const int VOCAB = 248320;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Tok {
  llama_model* m = nullptr; const llama_vocab* v = nullptr;
  explicit Tok(const char* path) {
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    auto p = llama_model_default_params(); p.vocab_only = true; p.n_gpu_layers = 0;
    m = llama_model_load_from_file(path, p); v = llama_model_get_vocab(m);
  }
  std::vector<int> enc(const std::string& s) { std::vector<int> t(s.size() + 16); int n = llama_tokenize(v, s.c_str(), s.size(), t.data(), t.size(), true, true); t.resize(std::max(n, 0)); return t; }
  std::string dec(int t) { char p[256]; int n = llama_token_to_piece(v, t, p, sizeof p, 0, true); return std::string(p, n > 0 ? n : 0); }
  std::string chat(const std::string& user) {
    llama_chat_message msg{"user", user.c_str()};
    std::vector<char> buf(user.size() * 2 + 4096);
    int n = llama_chat_apply_template(llama_model_chat_template(m, nullptr), &msg, 1, true, buf.data(), buf.size());
    return std::string(buf.data(), n) + "<think>\n\n</think>\n\n";
  }
};

static int row_capacity() { return 512; }
static int prefill_chunk() {
  const char* weights = getenv("H5_WEIGHTS");
  return weights && !strcmp(weights, "packed1") ? 8 : 512;
}

static bool bounded_int(const char* text, int lo, int hi, int& value) {
  if (!text || !*text) return false;
  char* end = nullptr; errno = 0;
  const long n = strtol(text, &end, 10);
  if (errno || *end || n < lo || n > hi) return false;
  value = (int)n; return true;
}

enum class ReferenceError { None, Open, Header, Geometry, Size, Tokens, Logits, Memory };
struct Reference {
  int prompt = 0;
  std::vector<int> tokens;
  std::vector<float> logits;
};
static const char* reference_error(ReferenceError error) {
  switch (error) {
    case ReferenceError::None: return "none";
    case ReferenceError::Open: return "reference file unavailable";
    case ReferenceError::Header: return "invalid reference header";
    case ReferenceError::Geometry: return "invalid reference geometry";
    case ReferenceError::Size: return "reference size does not match header";
    case ReferenceError::Tokens: return "invalid or truncated reference tokens";
    case ReferenceError::Logits: return "invalid or truncated reference logits";
    case ReferenceError::Memory: return "reference allocation failed";
  }
  abort();
}
static ReferenceError read_reference(const char* path, bool load_logits, Reference& out) {
  static_assert(sizeof(int) == 4 && sizeof(float) == 4, "reference format requires 32-bit values");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) return ReferenceError::Open;
  const auto bytes = file.tellg();
  file.seekg(0);
  int hdr[3];
  if (!file.read(reinterpret_cast<char*>(hdr), sizeof hdr)) return ReferenceError::Header;
  const int nt = hdr[0], np = hdr[1];
  if (np < 1 || nt <= np || nt > 65536 || hdr[2] != VOCAB) return ReferenceError::Geometry;
  const uint64_t nlogits = uint64_t(nt - np + 1) * VOCAB;
  const uint64_t expected = sizeof hdr + uint64_t(nt) * sizeof(int) + nlogits * sizeof(float);
  if (bytes < 0 || uint64_t(bytes) != expected) return ReferenceError::Size;
  try {
    Reference ref; ref.prompt = np; ref.tokens.resize(nt);
    if (!file.read(reinterpret_cast<char*>(ref.tokens.data()), nt * sizeof(int))) return ReferenceError::Tokens;
    for (int token : ref.tokens) if (token < 0 || token >= VOCAB) return ReferenceError::Tokens;
    if (load_logits) {
      ref.logits.resize(nlogits);
      if (!file.read(reinterpret_cast<char*>(ref.logits.data()), nlogits * sizeof(float))) return ReferenceError::Logits;
      for (float logit : ref.logits) if (!std::isfinite(logit)) return ReferenceError::Logits;
    }
    out = std::move(ref);
  } catch (const std::bad_alloc&) { return ReferenceError::Memory; }
  return ReferenceError::None;
}

enum class CompareError { None, Target, Nonfinite };
struct Cmp { double kld = 0, maxkld = 0, cos = 0, ref_nll = 0, got_nll = 0, maxerr = 0; int top1 = 0, n = 0; };
static CompareError compare(const float* ref, const float* got, Cmp& c, int target) {
  if (target < 0 || target >= VOCAB) return CompareError::Target;
  double mr = -std::numeric_limits<double>::infinity(), mg = mr, maxerr = 0; int ar = 0, ag = 0;
  for (int i = 0; i < VOCAB; i++) {
    if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) return CompareError::Nonfinite;
    maxerr = std::max(maxerr, fabs((double)ref[i] - got[i]));
    if (ref[i] > mr) { mr = ref[i]; ar = i; } if (got[i] > mg) { mg = got[i]; ag = i; }
  }
  double zr = 0, zg = 0, dot = 0, nr = 0, ng = 0;
  for (int i = 0; i < VOCAB; i++) { zr += exp(ref[i] - mr); zg += exp(got[i] - mg); dot += (double)ref[i] * got[i]; nr += (double)ref[i] * ref[i]; ng += (double)got[i] * got[i]; }
  double lzr = log(zr), lzg = log(zg), kl = 0;
  for (int i = 0; i < VOCAB; i++) { double lp = ref[i] - mr - lzr, lq = got[i] - mg - lzg; kl += exp(lp) * (lp - lq); }
  const double cosine = nr && ng ? dot / sqrt(nr * ng) : (nr == ng ? 1. : 0.);
  c.kld += kl; c.maxkld = std::max(c.maxkld, kl); c.cos += cosine; c.top1 += ar == ag; c.n++;
  c.ref_nll += mr + lzr - ref[target]; c.got_nll += mg + lzg - got[target];
  c.maxerr = std::max(c.maxerr, maxerr);
  return CompareError::None;
}
static void print_comparison(const char* path, const Cmp& c) {
  printf("%s: %d rows  KLD mean %.17g max %.17g  cos %.17g  top1 %d/%d  ref_ppl %.17g got_ppl %.17g max_logit_error %.17g\n",
         path, c.n, c.kld / c.n, c.maxkld, c.cos / c.n, c.top1, c.n, exp(c.ref_nll / c.n), exp(c.got_nll / c.n), c.maxerr);
}

#include "two_path_runner.h"

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: h5 check|decodebench|gen|bench MODEL ...\n"); return 1; }
  std::string mode = argv[1]; const char* model = argv[2];
  H5HeadMode head_mode = H5_HEAD_SELECTED;
  if (const char* selection = getenv("H5_HEAD_ROWS")) {
    if (h5::parse_head_mode(selection, head_mode) != H5_HEAD_OK) {
      fprintf(stderr, "H5_HEAD_ROWS must be all or demand\n"); return 1;
    }
  }
  if ((mode == "pathbench" || mode == "pathguard" || mode == "pathmatrix") &&
      getenv("H5_DRAFT_MARGIN") && atof(getenv("H5_DRAFT_MARGIN")) > 0.f) {
    fprintf(stderr, "Two-path tests require exact argmax: H5_DRAFT_MARGIN must be 0.\n"); return 1;
  }
  if (mode == "pathbench") return run_pathbench(argc, argv);
  if (mode == "pathguard") return run_pathguard(argc, argv);
  if (mode == "pathmatrix") return run_pathmatrix(argc, argv);
  if (mode == "check") {
    if (argc != 4) { fprintf(stderr, "check MODEL REF.bin\n"); return 1; }
    Reference ref;
    const auto error = read_reference(argv[3], true, ref);
    if (error != ReferenceError::None) { fprintf(stderr, "%s\n", reference_error(error)); return 1; }
    int CH = prefill_chunk();
    if (const char* chunk = getenv("H5_CHUNK"); chunk && !bounded_int(chunk, 1, row_capacity(), CH)) {
      fprintf(stderr, "H5_CHUNK must be an integer in 1..%d\n", row_capacity()); return 1;
    }
    const auto& toks = ref.tokens; const int nt = toks.size(), np = ref.prompt;
    void* e = h5_create(model, 2, nt + 64, row_capacity());
    if (!e) return 1;
    std::vector<int> slot(nt, 0), pos(nt), out(nt);
    for (int i = 0; i < nt; i++) pos[i] = i;
    std::vector<float> lg((size_t)std::min(CH, nt - 1) * VOCAB);
    auto compare_row = [&](int position, const float* got, Cmp& c) {
      const auto code = compare(ref.logits.data() + (size_t)(position - np + 1) * VOCAB, got, c, toks[position + 1]);
      if (code != CompareError::None) fprintf(stderr, "comparison at position %d: %s\n", position,
        code == CompareError::Target ? "invalid target token" : "nonfinite logit");
      return code == CompareError::None;
    };
    h5_reset(e, 0);
    Cmp a;
    for (int c0 = 0; c0 < nt - 1; c0 += CH) {
      const int n = std::min(CH, nt - 1 - c0);
      if (h5_step_logits(e, n, slot.data() + c0, toks.data() + c0, pos.data() + c0, out.data() + c0, lg.data())) return 1;
      for (int j = std::max(0, np - 1 - c0); j < n; j++)
        if (!compare_row(c0 + j, lg.data() + (size_t)j * VOCAB, a)) return 1;
    }
    print_comparison("prefill", a);
    h5_reset(e, 0);
    Cmp b;
    for (int c0 = 0; c0 < np; c0 += CH) {
      const int n = std::min(CH, np - c0);
      if (h5_step_logits(e, n, slot.data() + c0, toks.data() + c0, pos.data() + c0, out.data() + c0, lg.data())) return 1;
      if (c0 + n == np && !compare_row(np - 1, lg.data() + (size_t)(n - 1) * VOCAB, b)) return 1;
    }
    for (int i = np; i < nt - 1; i++) {
      if (h5_step_logits(e, 1, &slot[i], &toks[i], &pos[i], &out[i], lg.data())) return 1;
      if (!compare_row(i, lg.data(), b)) return 1;
    }
    print_comparison("decode", b);
    return 0;
  }
  if (mode == "gen") {
    Tok tk(model);
    std::string p = argv[3]; int n = atoi(argv[4]);
    bool chat = argc > 5 && !strcmp(argv[5], "--chat");
    auto toks = tk.enc(chat ? tk.chat(p) : p);
    void* e = h5_create(model, 1, 4096, 512);
    int np = toks.size(); std::vector<int> slot(np, 0), pos(np), out(np);
    for (int i = 0; i < np; i++) pos[i] = i;
    h5_reset(e, 0);
    double t0 = now_s();
    if (h5::step_head_rows(e, np, slot.data(), toks.data(), pos.data(), out.data(), head_mode, {np - 1})) return 1;
    double tp = now_s() - t0;
    int t = out[np - 1]; std::string text = tk.dec(t);
    t0 = now_s(); int ng = 1;
    for (int i = 0; i < n - 1; i++, ng++) {
      int s = 0, ps = np + i, o;
      h5_step(e, 1, &s, &t, &ps, &o); t = o; text += tk.dec(t);
      if (t == 248046 || t == 248044) break;
    }
    double tg = now_s() - t0;
    printf("%s\n--- prompt %d tok %.3f s, gen %d tok %.1f tok/s\n", text.c_str(), np, tp, ng, (ng - 1) / tg);
    return 0;
  }
  if (mode == "gemm") {
    void* e = h5_create(model, 1, 64, 1024);
    for (int i = 3; i < argc; i++) h5_gemm_bench(e, atoi(argv[i]), 5);
    return 0;
  }
  if (mode == "gsm") {   // h5 gsm MODEL PROMPTS.bin OUT.bin SLOTS [MAXNEW] [PREFILL_BUDGET]
    Tok tk(model);
    std::vector<std::string> prompts;
    { FILE* f = fopen(argv[3], "rb"); int n; while (fread(&n, 4, 1, f) == 1) { std::string x(n, 0); if (fread(&x[0], 1, n, f) != (size_t)n) break; prompts.push_back(x); } fclose(f); }
    const int S = atoi(argv[5]), maxnew = argc > 6 ? atoi(argv[6]) : 768, budget = argc > 7 ? atoi(argv[7]) : 256;
    const int nreq = prompts.size();
    std::vector<std::vector<int>> ptok(nreq), gen(nreq);
    int maxp = 0;
    for (int i = 0; i < nreq; i++) { ptok[i] = tk.enc(prompts[i]); maxp = std::max(maxp, (int)ptok[i].size()); }
    void* e = h5_create(model, S, maxp + maxnew + 8, S + budget);
    if (!e) return 1;
    struct Sl { int req = -1, done = 0; };   // done = prompt rows already fed
    std::vector<Sl> sl(S);
    int next = 0, finished = 0; long gen_tok = 0, steps = 0, rows_total = 0;
    std::vector<int> rs, rt, rp, ro, head_rows;
    double t0 = now_s(), tstep = 0;
    while (finished < nreq) {
      for (int s = 0; s < S; s++) if (sl[s].req < 0 && next < nreq) { sl[s].req = next++; sl[s].done = 0; h5_reset(e, s); }
      rs.clear(); rt.clear(); rp.clear(); head_rows.clear();
      int bud = budget;
      std::vector<int> last_row(S, -1);
      for (int s = 0; s < S; s++) {
        Sl& q = sl[s]; if (q.req < 0) continue;
        auto& P = ptok[q.req];
        if (q.done < (int)P.size()) {
          const int n = std::min((int)P.size() - q.done, bud); if (n <= 0) continue;
          for (int i = 0; i < n; i++) { rs.push_back(s); rt.push_back(P[q.done + i]); rp.push_back(q.done + i); }
          q.done += n; bud -= n;
          if (q.done == (int)P.size()) last_row[s] = rs.size() - 1;
        } else {
          auto& G = gen[q.req];
          rs.push_back(s); rt.push_back(G.back()); rp.push_back((int)P.size() + (int)G.size() - 1);
          last_row[s] = rs.size() - 1;
        }
      }
      for (int row : last_row) if (row >= 0) head_rows.push_back(row);
      ro.resize(rs.size());
      const double ts = now_s();
      if (h5::step_head_rows(e, rs.size(), rs.data(), rt.data(), rp.data(), ro.data(), head_mode, head_rows)) { fprintf(stderr, "step failed\n"); return 1; }
      tstep += now_s() - ts; steps++; rows_total += rs.size();
      for (int s = 0; s < S; s++) {
        if (last_row[s] < 0) continue;
        Sl& q = sl[s]; auto& G = gen[q.req];
        const int t = ro[last_row[s]];
        G.push_back(t); gen_tok++;
        if (t == 248046 || t == 248044 || (int)G.size() >= maxnew) { sl[s].req = -1; finished++; }
      }
      if (steps % 200 == 0) fprintf(stderr, "\r%d/%d done, %ld tokens, %.0f tok/s   ", finished, nreq, gen_tok, gen_tok / (now_s() - t0));
    }
    const double dt = now_s() - t0;
    if (getenv("H5_TOKDUMP")) { FILE* tf = fopen(getenv("H5_TOKDUMP"), "w"); for (int i = 0; i < nreq; i++) { for (int t : gen[i]) fprintf(tf, "%d ", t); fprintf(tf, "\n"); } fclose(tf); }
    FILE* f = fopen(argv[4], "wb");
    for (int i = 0; i < nreq; i++) {
      std::string x; for (int t : gen[i]) if (t != 248046 && t != 248044) x += tk.dec(t);
      int n = x.size(); fwrite(&n, 4, 1, f); fwrite(x.data(), 1, n, f);
    }
    fclose(f);
    printf("\nh5 gsm: %d prompts, %ld generated tokens in %.1f s = %.0f tok/s (step time %.1f s), %d slots, %ld steps, %.1f rows/step\n",
           nreq, gen_tok, dt, gen_tok / dt, tstep, S, steps, (double)rows_total / steps);
    return 0;
  }
  if (mode == "spec") {   // h5 spec MODEL DRAFTER PROMPTS.bin OUT.bin SLOTS [MAXNEW] [PREFILL_BUDGET]
    Tok tk(model);
    std::vector<std::string> prompts;
    { FILE* f = fopen(argv[4], "rb"); int n; while (fread(&n, 4, 1, f) == 1) { std::string x(n, 0); if (fread(&x[0], 1, n, f) != (size_t)n) break; prompts.push_back(x); } fclose(f); }
    const int S = atoi(argv[6]), maxnew = argc > 7 ? atoi(argv[7]) : 768, budget = argc > 8 ? atoi(argv[8]) : 256;
    const int nreq = prompts.size();
    std::vector<std::vector<int>> ptok(nreq), gen(nreq);
    // H5_PREFIX=FILE: a shared prefix, prefilled once into a template slot and cloned into every request's slot
    std::vector<int> pre;
    if (const char* pf = getenv("H5_PREFIX")) { FILE* f = fopen(pf, "rb"); std::string x; char buf[4096]; size_t n; while ((n = fread(buf, 1, sizeof buf, f)) > 0) x.append(buf, n); fclose(f); pre = tk.enc(x); }
    const int npre = pre.size();
    int maxp = 0;
    for (int i = 0; i < nreq; i++) {
      auto q = tk.enc(prompts[i]);
      ptok[i] = pre; ptok[i].insert(ptok[i].end(), q.begin(), q.end());
      maxp = std::max(maxp, (int)ptok[i].size());
    }
    const int TS = S;   // template slot index (extra slot)
    void* e = h5_create(model, S + (npre ? 1 : 0), maxp + maxnew + 16, std::max(S * 8 + budget, npre ? 512 : 0));
    if (!e || h5_draft_load(e, argv[3])) return 1;
    h5_capture(e, 1);
    if (npre) {   // prefill the prefix into the template slot, feeding the drafter as we go
      double tp = now_s();
      h5_reset(e, TS); h5_keep(e, TS, 0);
      for (int p0 = 0; p0 < npre; p0 += 512) {
        const int n = std::min(512, npre - p0);
        std::vector<int> rs_(n, TS), rp_(n), ro_(n);
        for (int i = 0; i < n; i++) rp_[i] = p0 + i;
        if (h5::step_head_rows(e, n, rs_.data(), pre.data() + p0, rp_.data(), ro_.data(), head_mode, {})) return 1;
        int sl_ = TS, ni_ = n, ir_ = 0, ip_ = p0, an_ = -1, st_ = 0; int dd[7];
        h5_draft(e, 1, &sl_, &ni_, &ir_, &ip_, &an_, &st_, dd);
      }
      fprintf(stderr, "prefix: %d tokens prefilled once in %.2f s\n", npre, now_s() - tp);
    }
    struct Sl { int req = -1, done = 0, len = 0, anchor = -1, nd = 7; int dr[7]; };
    const float cut = getenv("H5_DRAFT_CUT") ? atof(getenv("H5_DRAFT_CUT")) : 1.f;   // measured best on GSM8K-4k; -1 verifies all 7   // verify drafts while selector margin >= cut
    std::vector<Sl> sl(S);
    int next = 0, finished = 0; long gen_tok = 0, steps = 0, rows_total = 0, vsteps = 0, vacc = 0;
    std::vector<int> rs, rt, rp, ro, r0(S), nr(S), head_rows;
    std::vector<int> ds, dni, dir, dip, dan, dst, drafts(S * 7);
    double t0 = now_s(), tver = 0, tdr = 0, pure_t = 0; long pure_steps = 0, pure_tok = 0; double ctx_sum = 0; long ctx_n = 0;
    while (finished < nreq) {
      for (int s = 0; s < S; s++) if (sl[s].req < 0 && next < nreq) {
        sl[s] = Sl(); sl[s].req = next++; h5_reset(e, s);
        if (npre) { h5_copy_slot(e, TS, s, npre); sl[s].done = npre; }
      }
      rs.clear(); rt.clear(); rp.clear(); head_rows.clear();
      int bud = budget;
      for (int s = 0; s < S; s++) {
        Sl& q = sl[s]; r0[s] = rs.size(); nr[s] = 0; if (q.req < 0) continue;
        auto& P = ptok[q.req];
        if (q.anchor < 0) {   // prefill
          const int n = std::min((int)P.size() - q.done, bud); if (n <= 0) continue;
          h5_keep(e, s, 0);
          for (int i = 0; i < n; i++) { rs.push_back(s); rt.push_back(P[q.done + i]); rp.push_back(q.done + i); }
          nr[s] = n; bud -= n;
          h5::append_head_rows(head_rows, r0[s], n, q.done + n == (int)P.size()
            ? h5::HeadUse::FinishingPrefill : h5::HeadUse::IntermediatePrefill);
        } else {
          h5_keep(e, s, 1);
          rs.push_back(s); rt.push_back(q.anchor); rp.push_back(q.len);
          for (int k = 0; k < q.nd; k++) { rs.push_back(s); rt.push_back(q.dr[k]); rp.push_back(q.len + 1 + k); }
          nr[s] = 1 + q.nd;
          h5::append_head_rows(head_rows, r0[s], nr[s], h5::HeadUse::Verify);
        }
      }
      ro.resize(rs.size());
      double ts = now_s(); const double tstep0 = ts; const long gen0 = gen_tok;
      if (h5::step_head_rows(e, rs.size(), rs.data(), rt.data(), rp.data(), ro.data(), head_mode, head_rows)) { fprintf(stderr, "step failed\n"); return 1; }
      tver += now_s() - ts; steps++; rows_total += rs.size();
      bool pure = true; for (int q = 0; q < S; q++) if (nr[q] && sl[q].anchor < 0) pure = false;
      static const int trace = getenv("H5_TRACE_REQ") ? atoi(getenv("H5_TRACE_REQ")) : -1;
      for (int s = 0; s < S && trace >= 0; s++) if (sl[s].req == trace && nr[s]) {
        fprintf(stderr, "step %ld R=%zu slot %d gen %zu len %d rows:", steps, rs.size(), s, gen[trace].size(), sl[s].len);
        for (int k = 0; k < nr[s]; k++) fprintf(stderr, " %d", rt[r0[s] + k]);
        fprintf(stderr, " | out:"); for (int k = 0; k < nr[s] && k < 8; k++) fprintf(stderr, " %d", ro[r0[s] + k]);
        fprintf(stderr, " | batch:"); for (int q = 0; q < S; q++) if (nr[q]) fprintf(stderr, " %d:%d%s", q, nr[q], sl[q].anchor < 0 ? "p" : "");
        fprintf(stderr, "\n");
      }
      ds.clear(); dni.clear(); dir.clear(); dip.clear(); dan.clear(); dst.clear();
      std::vector<int> dslot;   // which slot each drafted block belongs to
      for (int s = 0; s < S; s++) {
        Sl& q = sl[s]; if (q.req < 0 || nr[s] == 0) continue;
        auto& P = ptok[q.req]; auto& G = gen[q.req];
        int ning, ipos0, an = -1;
        bool fin = false;
        if (q.anchor < 0) {
          ning = nr[s]; ipos0 = q.done; q.done += nr[s];
          if (q.done == (int)P.size()) {
            q.len = P.size(); an = ro[r0[s] + nr[s] - 1];
            G.push_back(an); gen_tok++;
            if (an == 248046 || an == 248044 || (int)G.size() >= maxnew) fin = true;
          }
        } else {
          int a = 0; while (a < q.nd && ro[r0[s] + a] == q.dr[a]) a++;
          ctx_sum += q.len; ctx_n++;
          vsteps++; vacc += a + 1;
          h5_accept(e, s, a + 1);
          ning = a + 1; ipos0 = q.len;
          for (int k = 0; k <= a && !fin; k++) {
            const int t = ro[r0[s] + k];   // drafts that matched, then the bonus
            G.push_back(t); gen_tok++;
            if (t == 248046 || t == 248044 || (int)G.size() >= maxnew) fin = true;
          }
          q.len += a + 1; an = ro[r0[s] + a];
        }
        if (fin) { q.req = -1; finished++; an = -1; }
        if (an >= 0) q.anchor = an;
        ds.push_back(s); dni.push_back(ning); dir.push_back(r0[s]); dip.push_back(ipos0); dan.push_back(an); dst.push_back(q.len);
        if (an >= 0) dslot.push_back(s);
      }
      ts = now_s();
      if (h5_draft(e, ds.size(), ds.data(), dni.data(), dir.data(), dip.data(), dan.data(), dst.data(), drafts.data())) { fprintf(stderr, "draft failed\n"); return 1; }
      tdr += now_s() - ts;
      if (pure) { pure_t += now_s() - tstep0; pure_steps++; pure_tok += gen_tok - gen0; }
      const float* cf = h5_draft_conf(e);
      for (size_t b = 0; b < dslot.size(); b++) {
        Sl& q = sl[dslot[b]];
        memcpy(q.dr, &drafts[b * 7], 7 * 4);
        q.nd = 7;
        if (cut >= 0.f) { q.nd = 0; while (q.nd < 7 && cf[b * 7 + q.nd] >= cut) q.nd++; }
      }
      if (steps % 50 == 0) fprintf(stderr, "\r%d/%d done, %ld tokens, %.0f tok/s, %.2f tok/verify   ", finished, nreq, gen_tok, gen_tok / (now_s() - t0), vsteps ? (double)vacc / vsteps : 0.);
    }
    const double dt = now_s() - t0;
    FILE* f = fopen(argv[5], "wb");
    for (int i = 0; i < nreq; i++) {
      std::string x; for (int t : gen[i]) if (t != 248046 && t != 248044) x += tk.dec(t);
      int n = x.size(); fwrite(&n, 4, 1, f); fwrite(x.data(), 1, n, f);
    }
    fclose(f);
    h5_prof_report(e);
    printf("\nh5 spec: %d prompts, %ld tokens in %.1f s = %.0f tok/s; %d slots, %ld steps, %.1f rows/step, %.3f tokens/verify; verify %.1f s, draft %.1f s\n",
           nreq, gen_tok, dt, gen_tok / dt, S, steps, (double)rows_total / steps, (double)vacc / vsteps, tver, tdr);
    printf("decode-only steps (no prefill rows): %ld steps, %ld tokens in %.1f s = %.0f tok/s, mean context %.0f tokens\n",
           pure_steps, pure_tok, pure_t, pure_tok / std::max(pure_t, 1e-9), ctx_sum / std::max(ctx_n, 1L));
    return 0;
  }
  if (mode == "specbench") {   // h5 specbench MODEL DRAFTER STREAMS CTX STEPS ACCEPT [ROWS [nodraft]]: steady-state drafted decode, forced acceptance
    // ROWS = verifier rows per stream (anchor + ROWS-1 proposals, default 8); nodraft skips the drafter (plain decode cost).
    const int S = atoi(argv[4]), C = atoi(argv[5]), steps = atoi(argv[6]), A = atoi(argv[7]);
    const int RW = argc > 8 ? atoi(argv[8]) : 8; const bool nodraft = argc > 9 && !strcmp(argv[9], "nodraft");
    if (RW < 1 || RW > 8 || A < 0 || A + 1 > RW) { fprintf(stderr, "specbench: need 1 <= ROWS <= 8 and ACCEPT < ROWS\n"); return 2; }
    void* e = h5_create(model, S, C + (steps + 2) * 8 + 16, S * 8);
    if (!e || (!nodraft && h5_draft_load(e, argv[3]))) return 1;
    h5_capture(e, nodraft ? 0 : 1);
    h5_pretune(e);   // reads the frozen H5_PLAN_READ plan (as the server does); without it every shape autotunes
    {   // the step must succeed: a failed step returns at once and would time as a fast step
      std::vector<int> s0(1, 0), t0(1, 1000), p0(1, C - 1), o0(1);
      if (h5_step(e, 1, s0.data(), t0.data(), p0.data(), o0.data())) { fprintf(stderr, "specbench: warm-up step failed\n"); return 1; }
      h5_reset(e, 0);
    }
    std::vector<int> len(S, C), anchor(S, 1000), dr(S * 7, 1000);
    std::vector<int> rs(S * RW), rt(S * RW), rp(S * RW), ro(S * RW), sl(S), ni(S), ir(S), ip(S), an(S), stt(S);
    for (int s = 0; s < S; s++) { h5_reset(e, s); h5_keep(e, s, nodraft ? 0 : 1); sl[s] = s; }
    double tv = 0, td = 0, t0 = 0;
    for (int it = -2; it < steps; it++) {
      if (it == 0) { h5_prof_report(e); t0 = now_s(); tv = td = 0; }
      for (int s = 0; s < S; s++)
        for (int k = 0; k < RW; k++) { rs[s * RW + k] = s; rt[s * RW + k] = k ? dr[s * 7 + k - 1] : anchor[s]; rp[s * RW + k] = len[s] + k; }
      double a = now_s();
      if (h5_step(e, S * RW, rs.data(), rt.data(), rp.data(), ro.data())) { fprintf(stderr, "specbench: step failed\n"); return 1; }
      double b = now_s();
      for (int s = 0; s < S; s++) {
        if (!nodraft) h5_accept(e, s, A + 1);
        ni[s] = A + 1; ir[s] = s * RW; ip[s] = len[s]; len[s] += A + 1; an[s] = ro[s * RW + A]; stt[s] = len[s]; anchor[s] = an[s];
      }
      if (!nodraft) h5_draft(e, S, sl.data(), ni.data(), ir.data(), ip.data(), an.data(), stt.data(), dr.data());
      double c = now_s();
      if (it >= 0) { tv += b - a; td += c - b; }
    }
    const double dt = (now_s() - t0) / steps;
    h5_prof_report(e);
    const double tps = S * (A + 1) / dt;
    printf("specbench: streams %d ctx %d rows/step %d forced tokens/step %d%s: step %.2f ms (verify %.2f, draft %.2f) -> %.0f tok/s, %.1f tok/s/stream, $%.4f/M\n",
           S, C, S * RW, A + 1, nodraft ? " nodraft" : "", dt * 1e3, tv / steps * 1e3, td / steps * 1e3, tps, tps / S, 0.69e6 / 3600 / tps);
    return 0;
  }
  if (mode == "draftdump") {   // h5 draftdump MODEL DRAFTER PROPOSALS STEPS "PROMPT": greedy decode, print the drafts and gaps after each round
    // Same drafter state across engines (P7 vs P15 + H5_DRAFT_NEST=8): proposals 1..7 must agree up to rounding.
    const int P = atoi(argv[4]), steps = atoi(argv[5]);
    Tok tk(model); auto T = tk.enc(tk.chat(argv[6])); const int np = T.size();
    if (np > 120) { fprintf(stderr, "draftdump: prompt must be <= 120 tokens\n"); return 2; }
    void* e = h5_create_draft(model, 1, np + (steps + 2) * (P + 1) + 64, P + 1 + 128, P);
    if (!e || h5_draft_load(e, argv[3])) return 1;
    h5_pretune(e);   // loads the frozen launch plan's shapes
    h5_capture(e, 1); h5_reset(e, 0); h5_keep(e, 0, 1);   // speculative-keep path, as specrows (frozen plan shapes)
    // Prefill in 8-row chunks: the frozen launch plan carries the verifier's row buckets, not arbitrary prompt widths.
    std::vector<int> rs(8, 0), rp(8), ro(8), dr(P);
    int s0 = 0, anchor = -1, len = np;
    for (int p0 = 0; p0 < np; p0 += 8) {
      int n = std::min(8, np - p0), ir = 0, ip = p0, st = 0, an = -1;
      for (int i = 0; i < n; i++) rp[i] = p0 + i;
      if (h5_step(e, n, rs.data(), T.data() + p0, rp.data(), ro.data())) { fprintf(stderr, "draftdump: prefill step failed\n"); return 1; }
      h5_accept(e, 0, n);
      if (p0 + n == np) { an = ro[n - 1]; st = np; anchor = an; }
      if (h5_draft(e, 1, &s0, &n, &ir, &ip, &an, &st, dr.data())) { fprintf(stderr, "draftdump: prefill draft failed\n"); return 1; }
    }
    int ni = 0, ir = 0, ip = 0;
    for (int it = 0; it < steps; it++) {
      const float* cf = h5_draft_conf(e);
      printf("round %d anchor %d drafts", it, anchor); for (int k = 0; k < P; k++) printf(" %d", dr[k]);
      printf(" gaps"); for (int k = 0; k < P; k++) printf(" %.3f", cf ? cf[k] : -1.f); printf("\n");
      // verify only the first 7 proposals so both engines follow the identical greedy trajectory
      const int R = 8; std::vector<int> vt(R), vs(R, 0), vp(R), vo(R);
      for (int k = 0; k < R; k++) { vt[k] = k ? dr[k - 1] : anchor; vp[k] = len + k; }
      h5_step(e, R, vs.data(), vt.data(), vp.data(), vo.data());
      int a = 0; while (a < 7 && vo[a] == dr[a]) a++;
      h5_accept(e, 0, a + 1);
      ni = a + 1; ir = 0; ip = len; len += a + 1; anchor = vo[a];
      h5_draft(e, 1, &s0, &ni, &ir, &ip, &anchor, &len, dr.data());
    }
    fflush(stdout); return 0;
  }
  if (mode == "specrows") {   // h5 specrows MODEL DRAFTER CTX STEPS PROPOSALS R1,R2,...: one stream, verify step of R rows + drafter block of PROPOSALS+1 rows
    const int C = atoi(argv[4]), steps = atoi(argv[5]), P = atoi(argv[6]);
    std::vector<int> Rs; for (const char* p = argv[7]; *p;) { Rs.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; }
    for (int R : Rs) if (R < 1 || R > P + 1) { fprintf(stderr, "specrows: rows must be 1..PROPOSALS+1\n"); return 2; }
    void* e = h5_create_draft(model, 1, C + (steps + 4) * (P + 1) * (int)Rs.size() + 64, P + 1 + 128, P);
    if (!e || h5_draft_load(e, argv[3])) return 1;
    h5_pretune(e);
    h5_capture(e, 1);
    int len = C, anchor = 1000, s0 = 0; std::vector<int> dr(P, 1000), rs(P + 1, 0), rt(P + 1), rp(P + 1), ro(P + 1);
    h5_reset(e, 0); h5_keep(e, 0, 1);
    for (int R : Rs) {
      double tv = 0, td = 0;
      for (int it = -3; it < steps; it++) {
        for (int k = 0; k < R; k++) { rt[k] = k ? dr[k - 1] : anchor; rp[k] = len + k; }
        const double a = now_s();
        h5_step(e, R, rs.data(), rt.data(), rp.data(), ro.data());
        const double b = now_s();
        h5_accept(e, 0, R);
        int ni = R, ir = 0, ip = len; len += R; anchor = ro[R - 1]; int st = len;
        h5_draft(e, 1, &s0, &ni, &ir, &ip, &anchor, &st, dr.data());
        const double c = now_s();
        if (it >= 0) { tv += b - a; td += c - b; }
      }
      printf("specrows: proposals %d rows %d ctx %d: verify %.3f ms draft %.3f ms round %.3f ms\n", P, R, len, tv / steps * 1e3, td / steps * 1e3, (tv + td) / steps * 1e3);
      fflush(stdout);
    }
    return 0;
  }
  if (mode == "spectest") {   // h5 spectest MODEL PROMPT N CORRUPT_PCT: oracle drafts with corruption must reproduce greedy
    Tok tk(model);
    auto P = tk.enc(tk.chat(argv[3])); const int n = atoi(argv[4]), corrupt = atoi(argv[5]);
    void* e = h5_create(model, 1, 4096, 512);
    const int np = P.size();
    std::vector<int> sl(np, 0), pos(np), out(np);
    for (int i = 0; i < np; i++) pos[i] = i;
    auto greedy = [&]() {
      h5_reset(e, 0); h5_keep(e, 0, 0);
      h5_step(e, np, sl.data(), P.data(), pos.data(), out.data());
      std::vector<int> g{out[np - 1]};
      while ((int)g.size() < n) { int s0 = 0, p = np + g.size() - 1, o; h5_step(e, 1, &s0, &g.back(), &p, &o); g.push_back(o); }
      return g;
    };
    auto ref = greedy();
    srand(1);
    h5_reset(e, 0); h5_keep(e, 0, 0);
    h5_step(e, np, sl.data(), P.data(), pos.data(), out.data());
    std::vector<int> g{out[np - 1]};
    h5_keep(e, 0, 1);
    long steps = 0, acc = 0;
    while ((int)g.size() < n) {
      const int base = np + g.size() - 1;   // position of the anchor g.back()
      std::vector<int> rows{g.back()};
      for (int k = 1; k < 8; k++) { int idx = g.size() - 1 + k; int t = idx < n ? ref[idx] : 0; if (rand() % 100 < corrupt) t = (t + 7) % 1000; rows.push_back(t); }
      std::vector<int> rs(8, 0), rp(8), ro(8);
      for (int k = 0; k < 8; k++) rp[k] = base + k;
      h5_step(e, 8, rs.data(), rows.data(), rp.data(), ro.data());
      int a = 0; while (a < 7 && ro[a] == rows[a + 1]) a++;
      if (getenv("SPEC_DEBUG")) { fprintf(stderr, "step %ld base %d a=%d  rows:", steps, base, a); for (int k = 0; k < 8; k++) fprintf(stderr, " %d", rows[k]); fprintf(stderr, "  out:"); for (int k = 0; k < 8; k++) fprintf(stderr, " %d", ro[k]); fprintf(stderr, "\n"); }
      for (int k = 0; k < a; k++) g.push_back(rows[k + 1]);
      g.push_back(ro[a]);
      h5_accept(e, 0, a + 1);
      steps++; acc += a;
    }
    g.resize(n);
    int same = 0; while (same < n && g[same] == ref[same]) same++;
    printf("spectest: %d tokens, %ld steps, %.2f accepted drafts/step, matches greedy for %d/%d tokens\n", n, steps, (double)acc / steps, same, n);
    if (same < n) { printf("ref:  "); for (int i = same; i < std::min(n, same + 8); i++) printf("%d ", ref[i]); printf("\nspec: "); for (int i = same; i < std::min(n, same + 8); i++) printf("%d ", g[i]); printf("\n"); }
    return 0;
  }
  if (mode == "decodebench") {
    if (argc != 6) { fprintf(stderr, "decodebench MODEL REF.bin STEPS REPEATS\n"); return 1; }
    int steps, reps;
    if (!bounded_int(argv[4], 1, 65536, steps) || !bounded_int(argv[5], 1, 100, reps)) {
      fprintf(stderr, "decodebench requires integer STEPS in 1..65536 and REPEATS in 1..100\n"); return 1;
    }
    Reference ref;
    const auto error = read_reference(argv[3], false, ref);
    if (error != ReferenceError::None) { fprintf(stderr, "%s\n", reference_error(error)); return 1; }
    const int nt = ref.tokens.size(), np = ref.prompt;
    if (steps > nt - np) { fprintf(stderr, "decodebench requires STEPS real continuation tokens\n"); return 1; }
    void* e = h5_create(model, 1, nt + 64, row_capacity());
    if (!e) return 1;
    std::vector<int> slots(np, 0), pos(np), out(np);
    std::vector<double> seconds;
    for (int i = 0; i < np; i++) pos[i] = i;
    for (int r = -1; r < reps; r++) {
      h5_reset(e, 0);
      for (int c = 0; c < np; c += prefill_chunk()) {
        const int n = std::min(prefill_chunk(), np - c);
        std::vector<int> heads;
        if (c + n == np) heads.push_back(n - 1);
        if (h5::step_head_rows(e, n, slots.data() + c, ref.tokens.data() + c, pos.data() + c, out.data() + c, head_mode, heads)) return 1;
      }
      const double t0 = now_s();
      for (int i = 0; i < steps; i++) {
        int s = 0, p = np + i, o;
        if (h5_step(e, 1, &s, &ref.tokens[p], &p, &o)) return 1;
      }
      const double elapsed = now_s() - t0;
      if (r >= 0) seconds.push_back(elapsed);
    }
    printf("{\"mode\":\"decodebench\",\"prefix_tokens\":%d,\"decode_tokens\":%d,\"seconds\":[", np, steps);
    for (size_t r = 0; r < seconds.size(); r++) printf("%s%.17g", r ? "," : "", seconds[r]);
    printf("]}\n");
    return 0;
  }
  if (mode == "bench") {
    const int S = atoi(argv[3]), C = atoi(argv[4]), steps = atoi(argv[5]);
    void* e = h5_create(model, S, C + steps + 8, std::max(S, 512));
    std::vector<int> slot(S), tok(S, 1000), pos(S, C), out(S);
    for (int s = 0; s < S; s++) { slot[s] = s; h5_reset(e, s); }
    // context is simulated: KV positions [0, C) hold zeros; attention still reads all of them
    h5_step(e, S, slot.data(), tok.data(), pos.data(), out.data());
    h5_step(e, S, slot.data(), tok.data(), pos.data(), out.data());
    h5_prof_report(e);
    double t0 = now_s();
    for (int i = 0; i < steps; i++) { for (int s = 0; s < S; s++) pos[s] = C + 1 + i; h5_step(e, S, slot.data(), tok.data(), pos.data(), out.data()); }
    double dt = (now_s() - t0) / steps;
    h5_prof_report(e);
    double tps = S / dt;
    printf("bench: streams %d ctx %d tokens/step 1: step %.2f ms, %.0f tok/s aggregate, %.1f tok/s/stream, $%.4f/M\n", S, C, dt * 1e3, tps, 1 / dt, 0.69e6 / 3600 / tps);
    return 0;
  }
  return 1;
}
