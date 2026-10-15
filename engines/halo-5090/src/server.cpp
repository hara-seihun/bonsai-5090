// h5-server: OpenAI-compatible HTTP server for halo-5090 (Ternary Bonsai 2 27B on one RTX 5090).
//
//   h5-server MODEL DRAFTER [--host 0.0.0.0] [--port 8000] [--slots 32] [--ctx 8192] [--prefill 512]
//
// Endpoints: GET /v1/models, GET /health, GET /metrics, POST /v1/chat/completions, POST /v1/completions
// (both with "stream": true as server-sent events). Decoding is greedy (temperature is accepted
// and ignored). Every step of the engine is one batched DFlash2-verified forward over all active
// slots: continuous batching with chunked prefill, speculative drafts verified by the target.
#include "llama.h"
#include "ggml-backend.h"
#include "chat.h"
#include "vision.h"
#include "model.h"
#include "two_path_policy.h"
#include "draft_contract.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "profile_trace.h"

using json = nlohmann::ordered_json;

extern "C" {
void* h5_create(const char* path, int slots, int ctx, int rows);
void* h5_create_draft(const char* path, int slots, int ctx, int rows, int proposals);
int h5_step(void* e, int R, const int* slot, const int* tok, const int* pos, int* out);
int h5_step_visual(void* e, int R, const int* slot, const int* tok, const int* pos, const int* rope, const float* visual, int* out);
void h5_reset(void* e, int slot);
void h5_keep(void* e, int slot, int flag);
void h5_accept(void* e, int slot, int n);
int h5_draft_load(void* e, const char* path);
void h5_capture(void* e, int on);
const float* h5_draft_conf(void* e);
int h5_draft(void* e, int ns, const int* slot, const int* n_ing, const int* ing_row0, const int* ing_pos0, const int* anchor, const int* start, int* drafts);
int h5_draft_two(void*, int, const int*, const int*, const int*, const int*, const int*, const int*, int*, int*, int*, int);
int h5_clone_slot(void*, int, int, int);
long h5_graph_stats(void* e, int which);
void h5_pretune(void* e);
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const int EOS1 = 248046, EOS2 = 248044;   // <|im_end|>, <|endoftext|>
static const char* MODEL_ID = "ternary-bonsai-2-27b";

struct Tok {
  llama_model* m = nullptr; const llama_vocab* v = nullptr;
  explicit Tok(const char* path, bool metadata = false) {
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    auto p = llama_model_default_params(); p.vocab_only = !metadata; p.no_alloc = false; p.n_gpu_layers = 0;
    m = llama_model_load_from_file(path, p); if (m) v = llama_model_get_vocab(m);
  }
  std::vector<int> enc(const std::string& s) {
    std::vector<int> t(s.size() + 16);
    int n = llama_tokenize(v, s.c_str(), s.size(), t.data(), t.size(), false, true);
    if (n < 0) { t.resize(-n); n = llama_tokenize(v, s.c_str(), s.size(), t.data(), t.size(), false, true); }
    t.resize(std::max(n, 0)); return t;
  }
  std::string piece(int t) { char p[256]; int n = llama_token_to_piece(v, t, p, sizeof p, 0, false); return std::string(p, n > 0 ? n : 0); }
};

// Bytes of s that form complete UTF-8 characters (the rest is held back until it completes).
static size_t utf8_complete(const std::string& s) {
  size_t i = s.size(), back = 0;
  while (i > 0 && back < 4) {
    unsigned char c = s[i - 1];
    if ((c & 0xC0) == 0x80) { i--; back++; continue; }
    int need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
    return (back + 1 >= (size_t)need) ? s.size() : i - 1;
  }
  return s.size();
}

struct Request {
  std::vector<int> prompt;
  h5::VisualPrompt visual;
  bool vision_speculation = false;
  bool has_image() const { return visual.image_tokens > 0; }
  bool use_draft() const { return !has_image() || vision_speculation; }
  long proposed_drafts = 0, accepted_drafts = 0, verify_rounds = 0, verifier_rows = 0;
  std::vector<long> prefix_histogram, position_proposed, position_reached, position_accepted, position_emitted;
  std::vector<long> position_reject_target_eos, position_reject_proposed_eos, position_reject_other;
  double native_decode_s = 0, draft_s = 0, clone_s = 0;
  long tree_rounds = 0, tree_rescues = 0, tree_fallback_rounds = 0;
  int max_new = 512;
  std::vector<std::string> stop;
  bool stream = false;
  // output, guarded by mu
  std::mutex mu; std::condition_variable cv;
  std::string text;            // emitted text so far
  size_t sent = 0;             // bytes already delivered to the client (streaming)
  bool done = false; std::string finish = "stop";
  int completion_tokens = 0;
  std::atomic<bool> cancel{false};
  double t_submit = 0, t_first = 0, t_done = 0, native_prefill_s = 0;
  std::string pending;         // decoded bytes not yet emitted (incomplete UTF-8 / possible stop prefix)
};

struct Scheduler {
  void* e = nullptr; Tok* tk = nullptr;
  int S, ctx, budget, proposals = h5::DRAFT_BASE_PROPOSALS;
  bool tree2 = false; // One private scratch slot S, never admitted as a request slot.
  std::mutex qmu; std::condition_variable qcv; std::deque<std::shared_ptr<Request>> queue;
  std::atomic<long> total_tokens{0}, total_steps{0}, total_req{0}; std::atomic<int> active{0};
  std::atomic<long long> busy_ns{0}, verify_ns{0}, draft_ns{0};
  std::atomic<long> verify_rounds{0}, accepted_drafts{0}, proposed_drafts{0};
  std::atomic<long> tree_rounds{0}, tree_rescues{0}, tree_fallback_rounds{0};
  std::atomic<long long> clone_ns{0};

  struct Sl {
    std::shared_ptr<Request> r; int done = 0, len = 0, anchor = -1, nd = 0; int dr[h5::DRAFT_MAX_PROPOSALS]; int ngen = 0; bool fin = false;
    int alt[7], fork = -1; bool alt_valid = false;
  };

  void submit(std::shared_ptr<Request> r) {
    if (!r->t_submit) r->t_submit = now_s();
    { std::lock_guard<std::mutex> l(qmu); queue.push_back(r); }
    qcv.notify_one();
  }

  // Append a generated token to the request; returns true when the request is finished.
  bool emit(Sl& q, int t, int verify_position = -1) {
    Request& r = *q.r;
    q.ngen++;
    if (t == EOS1 || t == EOS2) { finish(q, "stop"); return true; }
    std::string p = tk->piece(t);
    std::lock_guard<std::mutex> l(r.mu);
    if (r.t_first == 0) r.t_first = now_s();
    r.completion_tokens++;
    if (verify_position >= 0) r.position_emitted[verify_position]++;
    r.pending += p;
    // stop strings: cut at the first occurrence
    for (auto& s : r.stop) {
      if (s.empty()) continue;
      size_t at = (r.text + r.pending).find(s, r.text.size() > s.size() ? r.text.size() - s.size() : 0);
      if (at != std::string::npos) {
        std::string all = r.text + r.pending; all.resize(at);
        r.text = all; r.pending.clear(); r.done = true; r.finish = "stop"; r.t_done = now_s(); r.cv.notify_all(); return true;
      }
    }
    // hold back a possible stop-string prefix and incomplete UTF-8
    size_t keep = 0;
    for (auto& s : r.stop) {
      if (s.size() < 2) continue;
      for (size_t k = std::min(s.size() - 1, r.pending.size()); k > keep; k--)
        if (r.pending.compare(r.pending.size() - k, k, s, 0, k) == 0) { keep = k; break; }
    }
    size_t ok = utf8_complete(r.pending.substr(0, r.pending.size() - keep));
    r.text += r.pending.substr(0, ok); r.pending.erase(0, ok);
    r.cv.notify_all();
    if (q.ngen >= r.max_new) { r.text += r.pending.substr(0, utf8_complete(r.pending)); r.pending.clear(); r.done = true; r.finish = "length"; r.t_done = now_s(); r.cv.notify_all(); return true; }
    return false;
  }
  void finish(Sl& q, const char* why) {
    Request& r = *q.r;
    std::lock_guard<std::mutex> l(r.mu);
    r.text += r.pending.substr(0, utf8_complete(r.pending)); r.pending.clear(); r.done = true; r.finish = why; r.t_done = now_s(); r.cv.notify_all();
  }

  void run() {
    std::vector<Sl> sl(S);
    std::vector<int> rs, rt, rp, ro, r0(S), nr(S), drafts(S * proposals), alternatives(S * 7), forks(S);
    std::vector<int> ds, dni, dir, dip, dan, dst, dslot, rope;
    std::vector<float> visual;
    for (;;) {
      // admit
      {
        std::unique_lock<std::mutex> l(qmu);
        if (active.load() == 0) qcv.wait(l, [&] { return !queue.empty(); });
        for (int s = 0; s < S && !queue.empty(); s++)
          if (!sl[s].r) {
            auto r = queue.front(); queue.pop_front();
            if (r->cancel) { std::lock_guard<std::mutex> g(r->mu); r->done = true; r->finish = "cancelled"; r->cv.notify_all(); s--; continue; }
            sl[s] = Sl(); sl[s].r = r; h5_reset(e, s); active++;
          }
      }
      const double t_step0 = now_s();
      const int nact = active.load();
      // Few streams verify the configured proposal count; many cut at selector confidence.
      static const float cut_many = getenv("H5_DRAFT_CUT") ? atof(getenv("H5_DRAFT_CUT")) : 1.f;
      static const int all_below = getenv("H5_VERIFY_ALL_BELOW") ? atoi(getenv("H5_VERIFY_ALL_BELOW")) : 4;
      const float cut = nact <= all_below ? -1.f : cut_many;
      rs.clear(); rt.clear(); rp.clear();
      int bud = budget;
      int tree_s = -1, clone_back = -1;
      for (int s = 0; s < S; s++) {
        Sl& q = sl[s]; r0[s] = rs.size(); nr[s] = 0; if (!q.r) continue;
        if (q.r->cancel) { finish(q, "cancelled"); q.r.reset(); active--; continue; }
        auto& P = q.r->prompt;
        if (q.anchor < 0) {
          const int n = std::min((int)P.size() - q.done, bud); if (n <= 0) continue;
          h5_keep(e, s, 0);
          for (int i = 0; i < n; i++) { rs.push_back(s); rt.push_back(P[q.done + i]); rp.push_back(q.done + i); }
          nr[s] = n; bud -= n;
        } else {
          h5_keep(e, s, q.r->use_draft() ? 1 : 0);
          if (!q.r->use_draft()) q.nd = 0;
          rs.push_back(s); rt.push_back(q.anchor); rp.push_back(q.len);
          for (int k = 0; k < q.nd; k++) { rs.push_back(s); rt.push_back(q.dr[k]); rp.push_back(q.len + 1 + k); }
          nr[s] = 1 + q.nd;
          // Single-active text only. Multi-stream and every image retain the ordinary scheduler.
          if (tree2 && nact == 1 && !q.r->has_image() && q.nd == 7 && q.alt_valid) {
            const double tc = now_s();
            if (h5_clone_slot(e, s, S, q.len)) { fprintf(stderr, "tree clone failed\n"); abort(); }
            const double cs = now_s() - tc;
            q.r->clone_s += cs; clone_ns.fetch_add((long long)(cs * 1e9));
            tree_s = s;
            rs.push_back(S); rt.push_back(q.anchor); rp.push_back(q.len);
            for (int k = 0; k < 7; k++) { rs.push_back(S); rt.push_back(q.alt[k]); rp.push_back(q.len + 1 + k); }
          }
        }
      }
      if (rs.empty()) continue;
      ro.resize(rs.size());
      bool multimodal = false, embedding = false;
      for (size_t i = 0; i < rs.size(); i++) {
        const int owner = rs[i] == S ? tree_s : rs[i];
        multimodal |= sl[owner].r->has_image(); embedding |= rt[i] == -1;
      }
      int status; const double t_native0 = now_s();
      {
      h5trace::Range target_range("target-forward");
      if (multimodal) {
        rope.resize(4 * rs.size());
        if (embedding) visual.assign((size_t)rs.size() * h5::D, 0.f);
        for (size_t i = 0; i < rs.size(); i++) {
          const int owner = rs[i] == S ? tree_s : rs[i];
          const auto& r = *sl[owner].r; int p = rp[i];
          if (r.has_image() && p < (int)r.prompt.size()) {
            std::copy(r.visual.rope[p].begin(), r.visual.rope[p].end(), rope.begin() + 4 * i);
            const int er = r.visual.embedding_row[p];
            if (er >= 0) std::copy_n(r.visual.embeddings.data() + (size_t)er * h5::D, h5::D, visual.data() + i * h5::D);
          } else {
            const int position = r.has_image() ? r.visual.next_pos + p - r.prompt.size() : p;
            std::fill_n(rope.data() + 4 * i, 4, position);
          }
        }
        status = h5_step_visual(e, rs.size(), rs.data(), rt.data(), rp.data(), rope.data(), embedding ? visual.data() : nullptr, ro.data());
      } else status = h5_step(e, rs.size(), rs.data(), rt.data(), rp.data(), ro.data());
      }
      if (status) { fprintf(stderr, "step failed: %d\n", status); abort(); }
      const double native_s = now_s() - t_native0;
      verify_ns.fetch_add((long long)(native_s * 1e9), std::memory_order_relaxed);
      for (int s = 0; s < S; s++) if (sl[s].r && nr[s]) {
        if (sl[s].anchor < 0) sl[s].r->native_prefill_s += native_s;
        else sl[s].r->native_decode_s += native_s;
      }
      total_steps++;
      ds.clear(); dni.clear(); dir.clear(); dip.clear(); dan.clear(); dst.clear(); dslot.clear();
      for (int s = 0; s < S; s++) {
        Sl& q = sl[s]; if (!q.r || nr[s] == 0) continue;
        auto& P = q.r->prompt;
        const bool ordinary_image = q.r->has_image() && !q.r->use_draft();
        int ning, ipos0, an = -1, ingest_slot = s; bool fin = false;
        if (q.anchor < 0) {
          ning = nr[s]; ipos0 = q.done; q.done += nr[s];
          if (q.done == (int)P.size()) { q.len = P.size(); an = ro[r0[s] + nr[s] - 1]; total_tokens++; fin = emit(q, an); }
        } else {
          int a = h5::draft_accepted_prefix(ro.data() + r0[s], q.dr, q.nd);
          h5::PathChoice choice{h5::PathError::None, 0, a};
          const bool two = s == tree_s;
          if (tree2 && !two) { tree_fallback_rounds++; q.r->tree_fallback_rounds++; }
          if (two) {
            choice = h5::choose_two_path(rt.data() + r0[s], ro.data() + r0[s], 8, q.fork);
            if (choice.error != h5::PathError::None) { fprintf(stderr, "tree causal guard failed\n"); abort(); }
            a = choice.matched; tree_rounds++; q.r->tree_rounds++;
            if (choice.path) { tree_rescues++; q.r->tree_rescues++; ingest_slot = S; clone_back = s; }
          }
          const int proposed = two ? 14 : q.nd;
          q.r->proposed_drafts += proposed; q.r->accepted_drafts += a;
          if (q.r->use_draft()) {
            q.r->verify_rounds++; q.r->verifier_rows += two ? 16 : nr[s];
            q.r->prefix_histogram[a]++;
            if (!two && a < q.nd && !(a && h5::draft_eos(q.dr[a - 1]))) {
              const bool target_eos = h5::draft_eos(ro[r0[s] + a]);
              const bool proposed_eos = h5::draft_eos(q.dr[a]);
              if (target_eos) q.r->position_reject_target_eos[a]++;
              if (proposed_eos) q.r->position_reject_proposed_eos[a]++;
              if (!target_eos && !proposed_eos) q.r->position_reject_other[a]++;
            }
            if (!two) for (int p = 0; p < q.nd; p++) {
              q.r->position_proposed[p]++;
              if (p <= a && !(p == a && a && h5::draft_eos(q.dr[a - 1]))) q.r->position_reached[p]++;
              if (p < a) q.r->position_accepted[p]++;
            }
          }
          if (q.r->use_draft()) {
            verify_rounds++; accepted_drafts += a; proposed_drafts += proposed;
            h5_accept(e, ingest_slot, a + 1);
          }
          ning = a + 1; ipos0 = q.len;
          auto target = [&](int k) { return two ? h5::selected_target(ro.data() + r0[s], 8, q.fork, choice, k) : ro[r0[s] + k]; };
          for (int k = 0; k <= a && !fin; k++) { total_tokens++; fin = emit(q, target(k), k); }
          q.len += a + 1; an = target(a);
        }
        if (!fin && q.len + proposals + 2 >= ctx) { finish(q, "length"); fin = true; }
        if (fin) { q.r.reset(); active--; an = -1; }
        if (an >= 0) q.anchor = an;
        if (ordinary_image) continue;
        ds.push_back(ingest_slot); dni.push_back(ning); dir.push_back(r0[s] + (ingest_slot == S ? 8 : 0)); dip.push_back(ipos0); dan.push_back(an); dst.push_back(q.len);
        if (an >= 0) dslot.push_back(s);
      }
      const double t_draft0 = now_s();
      const bool two_proposal = tree2 && ds.size() == 1 && dslot.size() == 1 && active.load() == 1 && !sl[dslot[0]].r->has_image();
      int draft_status = 0;
      {
      h5trace::Range draft_range("drafter-forward");
      if (!ds.empty()) draft_status = two_proposal
        ? h5_draft_two(e, ds.size(), ds.data(), dni.data(), dir.data(), dip.data(), dan.data(), dst.data(), drafts.data(), alternatives.data(), forks.data(), 3)
        : h5_draft(e, ds.size(), ds.data(), dni.data(), dir.data(), dip.data(), dan.data(), dst.data(), drafts.data());
      }
      if (draft_status) { fprintf(stderr, "draft failed\n"); abort(); }
      const double draft_s = now_s() - t_draft0;
      draft_ns.fetch_add((long long)(draft_s * 1e9), std::memory_order_relaxed);
      for (int s : dslot) sl[s].r->draft_s += draft_s;
      // Ingest the selected captured rows before clone-back: capture belongs to scratch S.
      // Clone includes newly updated drafter ring AND pending causal state. Finished requests need no copy.
      if (clone_back >= 0 && sl[clone_back].r) {
        const double tc = now_s();
        if (h5_clone_slot(e, S, clone_back, sl[clone_back].len)) { fprintf(stderr, "tree clone-back failed\n"); abort(); }
        const double cs = now_s() - tc;
        sl[clone_back].r->clone_s += cs; clone_ns.fetch_add((long long)(cs * 1e9));
      }
      const float* cf = h5_draft_conf(e);
      for (size_t b = 0; b < dslot.size(); b++) {
        Sl& q = sl[dslot[b]];
        memcpy(q.dr, &drafts[b * proposals], proposals * sizeof(int));
        q.alt_valid = two_proposal;
        if (two_proposal) { memcpy(q.alt, &alternatives[b * 7], 7 * 4); q.fork = forks[b]; }
        q.nd = proposals;
        if (cut >= 0.f) { q.nd = 0; while (q.nd < proposals && cf[b * proposals + q.nd] >= cut) q.nd++; }
      }
      busy_ns.fetch_add((long long)((now_s() - t_step0) * 1e9), std::memory_order_relaxed);
    }
  }
};

static std::string jd(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }
static std::string rid() { static std::atomic<long> n{0}; char b[64]; snprintf(b, sizeof b, "h5-%lx-%ld", (long)time(nullptr), n++); return b; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: h5-server MODEL DRAFTER [--host H] [--port P] [--slots S] [--ctx C] [--prefill N] [--mmproj PATH] [--vision-gpu 0|1] [--image-max-tokens N] [--draft-proposals 7|16|32|64]\n"); return 1; }
  const bool tree2 = getenv("H5_TREE2") && atoi(getenv("H5_TREE2")) != 0;
  if (tree2 && getenv("H5_DRAFT_MARGIN") && atof(getenv("H5_DRAFT_MARGIN")) > 0.f) {
    fprintf(stderr, "H5_TREE2 requires exact argmax: H5_DRAFT_MARGIN must be 0.\n"); return 1;
  }
  std::string host = "0.0.0.0", mmproj; int port = 8000, S = 32, ctx = 8192, budget = 512;
  bool vision_gpu = false; int max_image_tokens = 1024, proposals = h5::DRAFT_BASE_PROPOSALS;
  if ((argc - 3) % 2) { fprintf(stderr, "every server option requires a value\n"); return 1; }
  for (int i = 3; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--host") host = argv[i + 1]; else if (k == "--port") port = atoi(argv[i + 1]);
    else if (k == "--slots") S = atoi(argv[i + 1]); else if (k == "--ctx") ctx = atoi(argv[i + 1]);
    else if (k == "--prefill") budget = atoi(argv[i + 1]);
    else if (k == "--mmproj") mmproj = argv[i + 1];
    else if (k == "--vision-gpu") vision_gpu = atoi(argv[i + 1]) != 0;
    else if (k == "--image-max-tokens") max_image_tokens = atoi(argv[i + 1]);
    else if (k == "--draft-proposals") proposals = h5::parse_proposals(argv[i + 1]);
    else { fprintf(stderr, "unknown server option: %s\n", k.c_str()); return 1; }
  }
  const auto draft_cfg = h5::draft_config(proposals, S, tree2, ctx, budget);
  if (draft_cfg.error != h5::DraftError::None) { fprintf(stderr, "%s\n", h5::draft_error(draft_cfg.error)); return 1; }
  if (proposals != 7 && !mmproj.empty()) { fprintf(stderr, "experimental long draft supports text only\n"); return 1; }
  fprintf(stderr, "draft proposals=%d query_rows=%d verifier_rows=%d (anchor+proposals; last output is greedy bonus) checkpoint_length_extrapolation=%s\n", proposals, draft_cfg.query_rows, draft_cfg.verifier_rows, proposals == 7 ? "false" : "true");
  llama_backend_init(); ggml_backend_load_all();
  Tok tk(argv[1], !mmproj.empty()); if (!tk.m) { fprintf(stderr, "tokenizer load failed\n"); return 1; }
  h5::Vision vision; std::string vision_error;
  if (!mmproj.empty() && !vision.init(mmproj.c_str(), tk.m, vision_gpu, max_image_tokens, vision_error)) {
    fprintf(stderr, "vision: %s\n", vision_error.c_str()); return 1;
  }
  Scheduler sch; sch.tk = &tk; sch.S = S; sch.ctx = ctx; sch.budget = budget;
  sch.tree2 = tree2; sch.proposals = proposals;
  // Scratch slot is private to the scheduler thread and never exposed/admitted.
  sch.e = h5_create_draft(argv[1], S + (sch.tree2 ? 1 : 0), ctx, draft_cfg.engine_rows, proposals);
  if (!sch.e || h5_draft_load(sch.e, argv[2])) return 1;
  h5_capture(sch.e, 1);
  h5_pretune(sch.e);
  std::thread worker([&] { sch.run(); });
  const double t_start = now_s();

  httplib::Server svr;
  svr.set_read_timeout(600); svr.set_write_timeout(600);
  svr.new_task_queue = [] { return new httplib::ThreadPool(256); };
  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(jd(json{{"status", "ok"}, {"vision", vision.enabled()}, {"vision_language_backend", "halo-native"}, {"draft_proposals", proposals}, {"draft_query_rows", proposals + 1}, {"verifier_rows", proposals + 1}, {"checkpoint_length_extrapolation", proposals != 7}, {"bonus_policy", "greedy"}}), "application/json");
  });
  std::mutex profile_mu;
  bool profile_active = false;
  const char* profile_env = getenv("H5_PROFILE_CONTROL");
  if (profile_env && std::string(profile_env) == "1") {
    auto profile_control = [&](bool start, httplib::Response& res) {
      std::lock_guard<std::mutex> guard(profile_mu);
      if (profile_active == start || sch.active.load() != 0) {
        res.status = 409;
        res.set_content("{\"error\":\"invalid profile transition or active request\"}", "application/json");
        return;
      }
#ifdef H5_PDLCLOCK
      const char* clock_dir = getenv("H5_PDL_OUTPUT");
      if (!clock_dir) { res.status = 500; res.set_content("{\"error\":\"H5_PDL_OUTPUT unset\"}", "application/json"); return; }
#endif
      const char* record_path=getenv("H5_REPLAY_WRITE");
      if(start && record_path) {
        int rc=h5_record_begin(record_path);
        if(rc) { res.status=500; res.set_content(jd(json{{"error","replay recorder failed"},{"code",rc}}),"application/json"); return; }
      }
      cudaError_t status = cudaSetDevice(0);
      if (status == cudaSuccess) status = start ? cudaProfilerStart() : cudaProfilerStop();
      if (status != cudaSuccess) {
        if(start && record_path) h5_record_end();
        res.status = 500;
        res.set_content(jd(json{{"error", cudaGetErrorString(status)}}), "application/json");
        return;
      }
      profile_active = start;
#ifdef H5_PDLCLOCK
      if (start) { h5_pdl_begin_kernels(); h5_pdl_begin_fp4(); h5_pdl_begin_draft(); }
      else {
        h5_pdl_end_kernels((std::string(clock_dir)+"/pdl-kernels.csv").c_str());
        h5_pdl_end_fp4((std::string(clock_dir)+"/pdl-fp4.csv").c_str());
        h5_pdl_end_draft((std::string(clock_dir)+"/pdl-draft.csv").c_str());
      }
#endif
      if(!start && record_path) {
        int rc=h5_record_end();
        if(rc) { res.status=500; res.set_content(jd(json{{"error","replay recorder close failed"},{"code",rc}}),"application/json"); return; }
      }
      res.set_content(jd(json{{"state", start ? "capturing" : "stopped"}}), "application/json");
    };
    svr.Post("/profile/start", [&, profile_control](const httplib::Request&, httplib::Response& res) { profile_control(true, res); });
    svr.Post("/profile/stop", [&, profile_control](const httplib::Request&, httplib::Response& res) { profile_control(false, res); });
  }
  svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
    json j = {{"object", "list"}, {"data", json::array({{{"id", MODEL_ID}, {"object", "model"}, {"owned_by", "halo-5090"}}})}};
    res.set_content(jd(j), "application/json");
  });
  svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
    json j = {{"draft_proposals", proposals}, {"draft_query_rows", proposals + 1}, {"verifier_rows_per_full_round", proposals + 1}, {"uptime_s", now_s() - t_start}, {"requests", sch.total_req.load()}, {"active", sch.active.load()},
              {"generated_tokens", sch.total_tokens.load()}, {"steps", sch.total_steps.load()}, {"busy_s", sch.busy_ns.load() * 1e-9},
              {"verify_s", sch.verify_ns.load() * 1e-9}, {"draft_s", sch.draft_ns.load() * 1e-9},
              {"verify_rounds", sch.verify_rounds.load()}, {"accepted_drafts", sch.accepted_drafts.load()},
              {"proposed_drafts", sch.proposed_drafts.load()},
              {"tree2_enabled", sch.tree2}, {"tree_rounds", sch.tree_rounds.load()}, {"tree_rescues", sch.tree_rescues.load()},
              {"tree_fallback_rounds", sch.tree_fallback_rounds.load()},
              {"clone_s", sch.clone_ns.load() * 1e-9},
              {"graphs_captured", h5_graph_stats(sch.e, 1)}, {"graph_replays", h5_graph_stats(sch.e, 0)}};
    res.set_content(jd(j), "application/json");
  });

  auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
    json body;
    try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content("{\"error\":{\"message\":\"invalid JSON\"}}", "application/json"); return; }
    auto r = std::make_shared<Request>(); r->t_submit = now_s();
    r->prefix_histogram.resize(proposals + 1); r->position_proposed.resize(proposals);
    r->position_reached.resize(proposals); r->position_accepted.resize(proposals); r->position_emitted.resize(proposals + 1);
    r->position_reject_target_eos.resize(proposals); r->position_reject_proposed_eos.resize(proposals); r->position_reject_other.resize(proposals);
    std::string prompt, error; std::vector<std::string> images;
    try {
      r->vision_speculation = body.value("vision_speculation", false);
      if (chat) {
        bool think = false;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].contains("enable_thinking")) think = body["chat_template_kwargs"]["enable_thinking"].get<bool>();
        if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string() && body["reasoning_effort"] != "none") think = true;
        if (!h5::chat_prompt(body.value("messages", json::array()), think, mtmd_default_marker(), prompt, images, error)) {
          res.status = 400; res.set_content(jd(json{{"error", {{"message", error}}}}), "application/json"); return;
        }
      } else {
        const auto& p = body.at("prompt");
        prompt = p.is_array() && !p.empty() ? p[0].get<std::string>() : p.get<std::string>();
      }
      if (!images.empty()) {
        if (!vision.prepare(prompt, images, ctx - 17, r->visual, error)) {
          res.status = 400; res.set_content(jd(json{{"error", {{"message", error}}}}), "application/json"); return;
        }
        r->prompt = r->visual.tokens;
      } else r->prompt = tk.enc(prompt);
    } catch (const json::exception& ex) {
      res.status = 400; res.set_content(jd(json{{"error", {{"message", std::string("invalid request: ") + ex.what()}}}}), "application/json"); return;
    }
    int mx = body.value("max_completion_tokens", body.value("max_tokens", 1024));
    const int context_reserve = std::max(16, proposals + 2);
    r->max_new = std::max(1, std::min(mx, ctx - (int)r->prompt.size() - context_reserve));
    if (r->prompt.empty() || (int)r->prompt.size() + context_reserve >= ctx) { res.status = 400; res.set_content(jd(json({{"error", {{"message", "prompt exceeds context " + std::to_string(ctx)}}}})), "application/json"); return; }
    if (body.contains("stop")) { auto& st = body["stop"]; if (st.is_string()) r->stop.push_back(st); else if (st.is_array()) for (auto& x : st) r->stop.push_back(x); }
    r->stream = body.value("stream", false);
    const bool usage_in_stream = body.contains("stream_options") && body["stream_options"].value("include_usage", false);
    const std::string id = rid(); const long created = time(nullptr);
    const std::string obj = chat ? "chat.completion" : "text_completion";
    sch.total_req++;
    sch.submit(r);
    auto usage = [&]() {
      json u = {{"prompt_tokens", (int)r->prompt.size()}, {"completion_tokens", r->completion_tokens}, {"total_tokens", (int)r->prompt.size() + r->completion_tokens}};
      return u;
    };
    if (!r->stream) {
      std::unique_lock<std::mutex> l(r->mu);
      r->cv.wait(l, [&] { return r->done; });
      json choice = chat ? json{{"index", 0}, {"message", {{"role", "assistant"}, {"content", r->text}}}, {"finish_reason", r->finish}}
                         : json{{"index", 0}, {"text", r->text}, {"finish_reason", r->finish}};
      json j = {{"id", id}, {"object", obj}, {"created", created}, {"model", MODEL_ID}, {"choices", json::array({choice})}, {"usage", usage()},
                {"timings", {{"image_tokens", r->visual.image_tokens}, {"image_encode_s", r->visual.encode_s}, {"native_prefill_s", r->native_prefill_s},
                             {"vision_speculation", r->has_image() && r->vision_speculation},
                             {"draft_proposals", proposals}, {"draft_query_rows", proposals + 1}, {"verifier_rows_per_full_round", proposals + 1},
                             {"checkpoint_length_extrapolation", proposals != 7}, {"bonus_policy", "greedy"},
                             {"verify_rounds", r->verify_rounds}, {"verifier_rows", r->verifier_rows},
                             {"prefix_histogram", r->prefix_histogram}, {"position_proposed", r->position_proposed},
                             {"position_reached", r->position_reached}, {"position_accepted", r->position_accepted}, {"position_emitted", r->position_emitted},
                             {"position_reject_target_eos", r->position_reject_target_eos}, {"position_reject_proposed_eos", r->position_reject_proposed_eos},
                             {"position_reject_other", r->position_reject_other},
                             {"proposed_drafts", r->proposed_drafts}, {"accepted_drafts", r->accepted_drafts},
                             {"verify_s", r->native_decode_s}, {"draft_s", r->draft_s},
                             {"tree_rounds", r->tree_rounds}, {"tree_rescues", r->tree_rescues}, {"clone_s", r->clone_s},
                             {"tree_fallback_rounds", r->tree_fallback_rounds},
                             {"ttft_s", r->t_first ? r->t_first - r->t_submit : 0.}, {"total_s", r->t_done - r->t_submit},
                             {"decode_tok_per_s", r->completion_tokens > 1 && r->t_done > r->t_first ? (r->completion_tokens - 1) / (r->t_done - r->t_first) : 0.}}}};
      res.set_content(jd(j), "application/json");
      return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream", [r, id, created, chat, usage_in_stream](size_t, httplib::DataSink& sink) {
      auto chunk = [&](const json& delta_or_text, const char* fin) {
        json choice = chat ? json{{"index", 0}, {"delta", delta_or_text}, {"finish_reason", fin ? json(fin) : json(nullptr)}}
                           : json{{"index", 0}, {"text", delta_or_text}, {"finish_reason", fin ? json(fin) : json(nullptr)}};
        json j = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created}, {"model", MODEL_ID}, {"choices", json::array({choice})}};
        std::string s = "data: " + jd(j) + "\n\n";
        return sink.write(s.data(), s.size());
      };
      if (chat && !chunk(json{{"role", "assistant"}, {"content", ""}}, nullptr)) { r->cancel = true; return false; }
      for (;;) {
        std::string piece; bool done; std::string fin;
        {
          std::unique_lock<std::mutex> l(r->mu);
          r->cv.wait_for(l, std::chrono::milliseconds(200), [&] { return r->done || r->text.size() > r->sent; });
          piece = r->text.substr(r->sent); r->sent = r->text.size(); done = r->done; fin = r->finish;
        }
        if (!piece.empty() && !chunk(chat ? json{{"content", piece}} : json(piece), nullptr)) { r->cancel = true; return false; }
        if (done) {
          chunk(chat ? json::object() : json(""), fin.c_str());
          if (usage_in_stream) {
            json j = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created}, {"model", MODEL_ID}, {"choices", json::array()},
                      {"usage", {{"prompt_tokens", (int)r->prompt.size()}, {"completion_tokens", r->completion_tokens}, {"total_tokens", (int)r->prompt.size() + r->completion_tokens}}}};
            std::string s = "data: " + jd(j) + "\n\n"; sink.write(s.data(), s.size());
          }
          std::string end = "data: [DONE]\n\n"; sink.write(end.data(), end.size());
          sink.done();
          return true;
        }
        if (!sink.is_writable()) { r->cancel = true; return false; }
      }
    });
  };
  svr.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, true); });
  svr.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, false); });
  fprintf(stderr, "h5-server listening on %s:%d (%d slots, ctx %d)\n", host.c_str(), port, S, ctx);
  svr.listen(host, port);
  return 0;
}
