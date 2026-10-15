#include "platform.h"
// h5-server: OpenAI-compatible HTTP server for halo-5090 (Bonsai checkpoints on one RTX 5090).
//
//   h5-server MODEL DRAFTER [--host 0.0.0.0] [--port 8000] [--slots 32] [--ctx 8192] [--prefill 512]
//
// Endpoints: GET /v1/models, GET /health, GET /metrics, POST /v1/chat/completions, POST /v1/completions
// (both with "stream": true as server-sent events). Greedy requests use DFlash2 verification;
// sampled single-slot requests use noise-coupled draft/target draws and causal prefix verification.
// Every step of the engine is one batched forward over all active
// slots: continuous batching with chunked prefill, speculative drafts verified by the target.
#include "llama.h"
#include "ggml-backend.h"
#include "chat.h"
#include "vision.h"
#include "model.h"
#include "two_path_policy.h"
#include "draft_contract.h"
#include "dspark_native.cuh"
#include "head_demand.h"
#include "commit_policy.h"
#include "serving_policy.h"
#include "batch_spec.h"
#include "prompt_lookup.h"
#include "relax_policy.h"
#include "lz_penalty.h"
#include "sampling.h"
#include <fstream>
#include <iterator>
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
int h5_row_stats(void* e, int R, h5::RowStat* out);
int h5_set_logit_adj(void* e, int R, const int* tok, const float* val);
int h5_set_draft_lz(void* e, int nb, const int* ctx, const int* len, const float* alpha);
int h5_step_visual(void* e, int R, const int* slot, const int* tok, const int* pos, const int* rope, const float* visual, int* out);
int h5_logits(void* e, int row, float* out);
void h5_reset(void* e, int slot);
int h5_copy_prefix(void* e, int src, int dst, int npos);
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

static json commitment_config(const h5::commit::Config& c, bool dspark) {
  const bool confidence = c.mode == h5::commit::Mode::Confidence || c.mode == h5::commit::Mode::Feedback;
  json result = {{"mode", h5::commit::name(c.mode)}, {"minimum", c.minimum}, {"maximum", c.maximum},
          {"confidence_gap_threshold", confidence && !dspark ? json(c.threshold) : json(nullptr)},
          {"confidence_probability_threshold", confidence && dspark ? json(c.threshold) : json(nullptr)},
          {"confidence_signal", confidence ? json(dspark ? "learned-acceptance-probability" : "draft-selector-best-minus-second-raw-score") : json(nullptr)},
          {"feedback_growth", c.mode == h5::commit::Mode::Feedback ? json(c.growth) : json(nullptr)},
          {"target_regret_observed", false}, {"selection_timing", "before-target-verification"}};
  if (c.mode == h5::commit::Mode::Expected) result["calibration"] = {c.cal_a, c.cal_b, c.cal_c};
  if (c.mode == h5::commit::Mode::Nested) {
    result["calibration"] = {c.cal_a, c.cal_b, c.cal_c};
    result["nested_full_accept_probability"] = c.threshold;
  }
  if (c.mode == h5::commit::Mode::Scheduled) {
    result["schedule"] = {{"id", h5::commit::SCHEDULE_ID}, {"seed", c.schedule_seed},
      {"counts", h5::commit::SCHEDULE_COUNTS}, {"array_length", h5::commit::SCHEDULE_LENGTH},
      {"algorithm", h5::commit::SCHEDULE_ALGORITHM}, {"sequence_hex", h5::commit::schedule_hex(c.schedule)},
      {"request_offset", h5::commit::SCHEDULE_OFFSET}, {"consumption", h5::commit::SCHEDULE_CONSUMPTION}};
  }
  return result;
}

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
  std::string piece(int t) {
    char p[256]; int n = llama_token_to_piece(v, t, p, sizeof p, 0, false);
    if (n > 0) return std::string(p, n);
    n = llama_token_to_piece(v, t, p, sizeof p, 0, true);
    const std::string marker(p, n > 0 ? n : 0);
    return marker == "<think>" || marker == "</think>" ? marker : std::string();
  }
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

// H5_NO_SPEC=1: plain decoding, one verifier row per stream and no drafter (not loaded).
static bool g_spec = true;

struct Request {
  std::vector<int> prompt;
  h5::VisualPrompt visual;
  bool vision_speculation = false;
  bool has_image() const { return visual.image_tokens > 0; }
  h5::sampling::Config sampling, draft_sampling;
  int draft_verify_limit = 0;
  std::vector<unsigned char> sampled_seen;
  float marker_lambda = 0.f;   // overthinking-marker penalty: request "marker_penalty", else the H5_MARKERS server lambda
  bool loop_close = false;   // H5_LOOP_CLOSE breaker for this request: request "loop_close", else on when configured
  bool lz_draft = false;  // drafter-side LZ for this request: request "lz_draft", else H5_LZ_DRAFT
  float lz_draft_alpha = -1.f;   // drafter strength: request "lz_draft_alpha", else the request's lz_penalty
  float lz_alpha = 0.f;   // LZ penalty strength for this request: request "lz_penalty", else the H5_LZ server alpha
  bool use_draft() const { return g_spec && draft_verify_limit > 0 && (!has_image() || vision_speculation); }
  long proposed_drafts = 0, accepted_drafts = 0, verify_rounds = 0, verifier_rows = 0;
  std::vector<long> prefix_histogram, position_proposed, position_reached, position_accepted, position_emitted;
  std::vector<long> position_reject_target_eos, position_reject_proposed_eos, position_reject_other;
  json commitment_rounds = json::array();
  std::string commitment_error;
  double native_decode_s = 0, draft_s = 0, clone_s = 0, sampling_s = 0;
  long tree_rounds = 0, tree_rescues = 0, tree_fallback_rounds = 0;
  long relax_offered = 0, relax_vetoed = 0, loop_closes = 0;
  int max_new = 0;
  std::vector<std::string> stop;
  bool stream = false;
  // output, guarded by mu
  std::mutex mu; std::condition_variable cv;
  std::string text;            // emitted text so far
  size_t sent = 0;             // bytes already delivered to the client (streaming)
  bool done = false, settled = false; std::string finish = "stop";
  int completion_tokens = 0;
  std::atomic<bool> cancel{false};
  double t_submit = 0, t_first = 0, t_done = 0, native_prefill_s = 0;
  std::string pending;         // decoded bytes not yet emitted (incomplete UTF-8 / possible stop prefix)
};

struct Scheduler {
  void* e = nullptr; Tok* tk = nullptr;
  int S, ctx, budget, proposals = h5::DRAFT_BASE_PROPOSALS;
  float cut_many;
  int all_below;
  h5::commit::Config commitment;
  H5HeadMode head_mode = H5_HEAD_SELECTED;
  bool tree2 = false; // One private scratch slot S, never admitted as a request slot.
  int lookup_extra_max = 0;   // H5_LOOKUP_EXTRA: 0 = prompt lookup off
  bool lookup_prompt_only = false;   // H5_LOOKUP_SCOPE=prompt
  int lookup_min = 0;                // H5_LOOKUP_MIN
  std::atomic<long> lookup_rounds{0}, lookup_proposed{0}, lookup_accepted{0};
  h5::relax::Policy relax;          // H5_RELAX_POLICY (relax_policy.h); inactive = plain margin
  float relax_margin = 0.f;
  std::vector<char> digit_table;
  h5::relax::Counters relax_counters;
  std::vector<h5::RowStat> rstat;
  bool relax_stats_env = false;
  h5::dry::Config dry;              // H5_DRY: repetition penalty on every verify row (needs H5_LOGIT_ADJ)
  bool lz_on = false; h5::lz::Config lz{};   // H5_LZ: LZ penalty on every verify row (lz_penalty.h)
  bool lz_draft = false;                      // H5_LZ_DRAFT=on: the drafter's selector applies it too
  std::vector<int> lzd_ctx, lzd_len; std::vector<float> lzd_alpha;
  int adj_k = 0, dry_k = 0;         // engine slots per row; DRY's share (the rest holds up to lz.window LZ entries)
  h5::lz::Scratch lz_scratch{h5::VOCAB};
  h5::adj::Row adj_row{h5::VOCAB};
  std::vector<int> dry_tok; std::vector<float> dry_val;
  std::atomic<long long> adjust_ns{0};
  std::vector<int> markers; float marker_lambda = 0.f;   // H5_MARKERS: fixed penalty on overthinking markers in <think>
  bool adjusting() const { return dry.active || lz_on || !markers.empty(); }
  std::vector<int> adj_tok; std::vector<float> adj_val;   // H5_RELAX_STATS set: row statistics are read every step (commit trace gets greedy/gap)
  h5::serving::Config serving;
  h5::bspec::Policy bspec;
  FILE* round_trace = nullptr;   // H5_ROUND_TRACE=FILE: one line per fixed-policy verify round (calibration)
  std::atomic<long> verifier_rows_total{0};
  int cursor = 0;
  std::vector<int> prefix;
  double prefix_build_s = 0;
  bool prefix_checked = false;
  long prefix_target_changes = 0, prefix_draft_changes = 0;
  std::atomic<long> prefix_hits{0}, prefix_misses{0}, prefill_rows{0}, cancelled{0};
  std::mutex qmu; std::condition_variable qcv; std::deque<std::shared_ptr<Request>> queue;
  std::atomic<long> total_tokens{0}, total_steps{0}, total_req{0}; std::atomic<int> active{0};
  std::atomic<long long> busy_ns{0}, verify_ns{0}, draft_ns{0};
  std::atomic<long> verify_rounds{0}, accepted_drafts{0}, proposed_drafts{0};
  std::atomic<long> tree_rounds{0}, tree_rescues{0}, tree_fallback_rounds{0};
  std::atomic<long long> clone_ns{0};

  struct Sl {
    std::shared_ptr<Request> r; int done = 0, len = 0, anchor = -1, nd = 0; int dr[h5::DRAFT_MAX_PROPOSALS]; float gaps[h5::DRAFT_MAX_PROPOSALS]; int ngen = 0; bool fin = false;
    int alt[7], fork = -1; bool alt_valid = false;
    h5::lookup::Index lookup;   // prompt + emitted tokens (text requests), for the agreement extension
    int lookup_extra = 0;       // proposals appended from the lookup this round
    h5::relax::State relax_state;
    std::vector<int> seq;       // prompt + emitted tokens (H5_DRY / H5_LZ only)
    bool in_think = false;      // a <think> block is open at the end of seq (H5_DRY_SCOPE / H5_LZ_SCOPE=think)
    std::vector<int> out;       // emitted tokens (H5_LOOP_CLOSE only)
    bool lc_think = false;      // a <think> block is open (H5_LOOP_CLOSE)
    void limit_drafts(int count) {
      if (count < 0) { fprintf(stderr, "negative verifier limit\n"); abort(); }
      const int native_prefix = nd - lookup_extra;
      nd = std::min(nd, count);
      lookup_extra = std::max(0, nd - native_prefix);
    }
    double last_prefill = 0;
    h5::commit::State commitment_state{0};
    h5::commit::Selection commitment_selection{h5::commit::Error::None, 0, 0, -1, 0.f, h5::commit::Trigger::FullBlock};
  };

  int think_open = -1, think_close = -1;
  h5::loopclose::Config loop_close;
  std::atomic<long> loop_closes{0};
  void dry_push(Sl& q, int t) { q.seq.push_back(t); if (t == think_open) q.in_think = true; else if (t == think_close) q.in_think = false; }

  void submit(std::shared_ptr<Request> r) {
    if (!r->t_submit) r->t_submit = now_s();
    { std::lock_guard<std::mutex> l(qmu); queue.push_back(r); }
    qcv.notify_one();
  }

  // Append a generated token to the request; returns true when the request is finished.
  bool emit(Sl& q, int t, int verify_position = -1) {
    Request& r = *q.r;
    q.ngen++;
    if (r.sampling.ordinary()) r.sampled_seen[t] = 1;
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

  void settle(Sl& q) {
    std::lock_guard<std::mutex> l(q.r->mu);
    q.r->settled = true;
    q.r->cv.notify_all();
  }

  bool build_prefix() {
    std::ifstream in(serving.prefix_file, std::ios::binary);
    if (!in) { fprintf(stderr, "prefix file unavailable: %s\n", serving.prefix_file.c_str()); return false; }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) { fprintf(stderr, "prefix file read failed\n"); return false; }
    prefix = tk->enc(text);
    if (prefix.empty() || prefix.size() + std::max(16, proposals + 2) >= (size_t)ctx) {
      fprintf(stderr, "prefix must be nonempty and leave context for continuation\n"); return false;
    }
    const double begin = now_s();
    const int donor = serving.template_slot, quantum = std::min(h5::serving::QUANTUM, budget);
    h5_reset(e, donor); h5_keep(e, donor, 0);
    for (int p = 0; p < (int)prefix.size(); p += quantum) {
      int n = std::min(quantum, (int)prefix.size() - p);
      std::vector<int> slots(n, donor), positions(n), out(n), draft(proposals);
      for (int i = 0; i < n; ++i) positions[i] = p + i;
      if (h5::step_head_rows(e, n, slots.data(), prefix.data() + p, positions.data(), out.data(), head_mode, {})) return false;
      int row = 0, anchor = -1, start = 0;
      if (h5_draft(e, 1, &donor, &n, &row, &p, &anchor, &start, draft.data())) return false;
    }
    prefix_build_s = now_s() - begin;
    return true;
  }

  bool check_prefix() {
    auto suffix = tk->enc("\nQuestion: What is 17 plus 25? Answer:");
    if (suffix.empty() || prefix.size() + suffix.size() + proposals + 10 >= (size_t)ctx) return false;
    const int donor = serving.template_slot, quantum = std::min(h5::serving::QUANTUM, budget);
    if (h5_copy_prefix(e, donor, donor, prefix.size()) != -1 || h5_copy_prefix(e, donor, 0, ctx + 1) != -1) return false;
    for (int reuse = 0; reuse < 2; ++reuse) {
      h5_reset(e, 0); h5_reset(e, 1);
      if (h5_copy_prefix(e, donor, 0, prefix.size())) return false;
      h5_keep(e, 1, 0);
      for (int p = 0; p < (int)prefix.size(); p += quantum) {
        int n = std::min(quantum, (int)prefix.size() - p), s = 1, row = 0, anchor = -1, start = 0;
        std::vector<int> slots(n, s), positions(n), out(n), draft(proposals);
        for (int i = 0; i < n; ++i) positions[i] = p + i;
        if (h5_step(e, n, slots.data(), prefix.data() + p, positions.data(), out.data()) ||
            h5_draft(e, 1, &s, &n, &row, &p, &anchor, &start, draft.data())) return false;
      }
      int anchors[2]; std::vector<int> draft[2] = {std::vector<int>(proposals), std::vector<int>(proposals)};
      auto step = [&](int s, const std::vector<int>& tokens, int p, bool keep, int ingest) {
        h5_keep(e, s, keep);
        std::vector<int> slots(tokens.size(), s), positions(tokens.size()), out(tokens.size());
        for (int i = 0; i < (int)tokens.size(); ++i) positions[i] = p + i;
        if (h5_step(e, tokens.size(), slots.data(), tokens.data(), positions.data(), out.data())) return false;
        if (keep) h5_accept(e, s, ingest);
        int row = 0, anchor = out[ingest - 1], start = p + ingest;
        if (h5_draft(e, 1, &s, &ingest, &row, &p, &anchor, &start, draft[s].data())) return false;
        anchors[s] = anchor; return true;
      };
      for (int s = 0; s < 2; ++s) if (!step(s, suffix, prefix.size(), false, suffix.size())) return false;
      for (int k = 0, p = prefix.size() + suffix.size(); k < 8; ++k, ++p) {
        prefix_target_changes += anchors[0] != anchors[1];
        for (int i = 0; i < proposals; ++i) prefix_draft_changes += draft[0][i] != draft[1][i];
        for (int s = 0; s < 2; ++s) {
          std::vector<int> tokens{anchors[s]};
          for (int i = 0; i < proposals; ++i) tokens.push_back((draft[s][i] + 1) % 248000);
          if (!step(s, tokens, p, true, 1)) return false;
          if (s == 0 && h5_copy_prefix(e, 0, 1, p) != -2) return false;
        }
      }
      prefix_target_changes += anchors[0] != anchors[1];
      for (int i = 0; i < proposals; ++i) prefix_draft_changes += draft[0][i] != draft[1][i];
    }
    h5_reset(e, 0); h5_reset(e, 1); prefix_checked = true; return true;
  }

  void run() {
    std::vector<Sl> sl(S);
    std::vector<int> rs, rt, rp, ro, r0(S), nr(S), hstart(S), drafts(S * proposals), alternatives(S * 7), forks(S);
    std::vector<int> ds, dni, dir, dip, dan, dst, dslot, rope, head_rows;
    std::vector<float> visual;
    for (;;) {
      // admit
      {
        std::unique_lock<std::mutex> l(qmu);
        if (active.load() == 0) qcv.wait(l, [&] { return !queue.empty(); });
        for (int s = 0; s < S && !queue.empty(); s++)
          if (!sl[s].r) {
            auto r = queue.front(); queue.pop_front();
            if (r->cancel) { std::lock_guard<std::mutex> g(r->mu); r->done = r->settled = true; r->finish = "cancelled"; r->t_done = now_s(); cancelled++; r->cv.notify_all(); s--; continue; }
            sl[s] = Sl(); sl[s].r = r; sl[s].last_prefill = now_s();
            if (lookup_extra_max && !r->has_image()) {
              sl[s].lookup = h5::lookup::Index(); sl[s].lookup.configure(lookup_prompt_only, lookup_min);
              sl[s].lookup.append(r->prompt.data(), r->prompt.size()); sl[s].lookup.freeze();
            }
            h5::relax::seed(relax, sl[s].relax_state, r->prompt.data(), r->prompt.size());
            if (loop_close.active) {
              sl[s].out.clear(); sl[s].lc_think = false;
              for (int t : r->prompt) { if (t == think_open) sl[s].lc_think = true; else if (t == think_close) sl[s].lc_think = false; }
            }
            if (adjusting()) {
              sl[s].seq = r->prompt; sl[s].in_think = false;
              for (int t : r->prompt) { if (t == think_open) sl[s].in_think = true; else if (t == think_close) sl[s].in_think = false; }
            }
            const int schedule_offset = commitment.mode == h5::commit::Mode::Scheduled
              ? h5::commit::prompt_schedule_offset(commitment, r->prompt.data(), r->prompt.size()) : 0;
            sl[s].commitment_state = h5::commit::initial(commitment, schedule_offset); h5_reset(e, s);
            if (!prefix.empty()) {
              if (h5::serving::prefix_match(prefix, r->prompt, r->has_image())) {
                if (h5_copy_prefix(e, serving.template_slot, s, prefix.size())) { fprintf(stderr, "prefix copy failed\n"); abort(); }
                sl[s].done = prefix.size(); prefix_hits++;
              } else prefix_misses++;
            }
            active++;
          }
      }
      const double t_step0 = now_s();
      const int nact = active.load();
      const float cut = nact <= all_below ? -1.f : cut_many;
      rs.clear(); rt.clear(); rp.clear(); head_rows.clear();
      int bud = budget;
      std::vector<int> allocation;
      if (serving.fair) {
        std::vector<h5::serving::Prefill> items;
        for (int s = 0; s < S; ++s) if (sl[s].r && sl[s].anchor < 0 && !sl[s].r->cancel)
          items.push_back({s, (int)sl[s].r->prompt.size() - sl[s].done, t_step0 - sl[s].last_prefill});
        auto result = h5::serving::allocate(std::move(items), S, budget, cursor);
        if (result.error != h5::serving::Error::None) { fprintf(stderr, "%s\n", h5::serving::error(result.error)); abort(); }
        allocation = std::move(result.rows); cursor = (cursor + 1) % S;
      }
      int tree_s = -1, clone_back = -1;
      for (int s = 0; s < S; s++) {
        Sl& q = sl[s]; r0[s] = rs.size(); nr[s] = 0; hstart[s] = head_rows.size(); if (!q.r) continue;
        if (q.r->cancel) { finish(q, "cancelled"); settle(q); q.r.reset(); h5_reset(e, s); active--; cancelled++; continue; }
        auto& P = q.r->prompt;
        if (q.anchor < 0) {
          const int n = serving.fair ? allocation[s] : std::min((int)P.size() - q.done, bud); if (n <= 0) continue;
          q.last_prefill = t_step0; prefill_rows += n;
          h5_keep(e, s, 0);
          for (int i = 0; i < n; i++) { rs.push_back(s); rt.push_back(P[q.done + i]); rp.push_back(q.done + i); }
          nr[s] = n; bud -= n;
          h5::append_head_rows(head_rows, r0[s], n, q.done + n == (int)P.size()
            ? h5::HeadUse::FinishingPrefill : h5::HeadUse::IntermediatePrefill);
        } else {
          h5_keep(e, s, q.r->use_draft() ? 1 : 0);
          if (!q.r->use_draft()) q.nd = 0;
          q.limit_drafts(std::max(0, ctx - q.len - 1));
          rs.push_back(s); rt.push_back(q.anchor); rp.push_back(q.len);
          for (int k = 0; k < q.nd; k++) { rs.push_back(s); rt.push_back(q.dr[k]); rp.push_back(q.len + 1 + k); }
          nr[s] = 1 + q.nd;
          h5::append_head_rows(head_rows, r0[s], nr[s], h5::HeadUse::Verify);
          // Single-active text only. Multi-stream and every image retain the ordinary scheduler.
          if (tree2 && nact == 1 && !q.r->has_image() && q.nd == 7 && q.alt_valid) {
            const double tc = now_s();
            if (h5_clone_slot(e, s, S, q.len)) { fprintf(stderr, "tree clone failed\n"); abort(); }
            const double cs = now_s() - tc;
            q.r->clone_s += cs; clone_ns.fetch_add((long long)(cs * 1e9));
            tree_s = s;
            rs.push_back(S); rt.push_back(q.anchor); rp.push_back(q.len);
            for (int k = 0; k < 7; k++) { rs.push_back(S); rt.push_back(q.alt[k]); rp.push_back(q.len + 1 + k); }
            h5::append_head_rows(head_rows, r0[s] + nr[s], 8, h5::HeadUse::Verify);
          }
        }
      }
      if (rs.empty()) continue;
      ro.resize(rs.size());
      if (adjusting()) {
        const double t_adj0 = now_s();
        adj_tok.assign(rs.size() * adj_k, -1); adj_val.assign(rs.size() * adj_k, 0.f);
        // Row k of a verify group sees the committed sequence plus the round's proposals 0..k-1.
        auto build = [&](Sl& q, int row, const int* ext, int k) {
          const long prompt = (long)q.r->prompt.size();
          bool think = q.in_think;
          for (int j = 0; j < k; ++j) { if (ext[j] == think_open) think = true; else if (ext[j] == think_close) think = false; }
          adj_row.begin();
          if (dry.active && (dry.scope != h5::dry::Scope::Think || q.in_think)) {
            const long start = dry.scope == h5::dry::Scope::All ? 0 : prompt;
            h5::dry::row(dry, q.seq, ext, k, dry_k, dry_tok.data(), dry_val.data(), start);
            for (int j = 0; j < dry_k; ++j) if (dry_tok[j] >= 0) adj_row.add(dry_tok[j], dry_val[j]);
          }
          if (lz_on && q.r->lz_alpha > 0.f && (lz.scope == h5::lz::Scope::Output || think)) {
            h5::lz::Config rc = lz; rc.alpha = q.r->lz_alpha;
            h5::lz::row(rc, q.seq.data() + prompt, (long)q.seq.size() - prompt, ext, k, lz_scratch,
                        [&](int t, float v) { adj_row.add(t, v); });
          }
          if (think && q.r->marker_lambda > 0.f) for (int t : markers) adj_row.add(t, -q.r->marker_lambda);
          if (!adj_row.write(adj_k, &adj_tok[(size_t)row * adj_k], &adj_val[(size_t)row * adj_k])) {
            fprintf(stderr, "logit adjustment row holds %zu tokens > H5_LOGIT_ADJ=%d\n", adj_row.size(), adj_k); abort();
          }
        };
        for (int s = 0; s < S; s++) {
          Sl& q = sl[s]; if (!q.r || !nr[s] || q.r->has_image()) continue;
          if (q.anchor < 0) {
            if (q.done + nr[s] == (int)q.r->prompt.size()) build(q, r0[s] + nr[s] - 1, nullptr, 0);
          } else for (int k = 0; k < nr[s]; ++k) build(q, r0[s] + k, q.dr, k);
        }
        if (h5_set_logit_adj(e, rs.size(), adj_tok.data(), adj_val.data())) { fprintf(stderr, "logit adjustment unavailable\n"); abort(); }
        adjust_ns.fetch_add((long long)((now_s() - t_adj0) * 1e9), std::memory_order_relaxed);
      }
      bool multimodal = false, embedding = false;
      for (size_t i = 0; i < rs.size(); i++) {
        const int owner = rs[i] == S ? tree_s : rs[i];
        multimodal |= sl[owner].r->has_image(); embedding |= rt[i] == -1;
      }
      const int nh = head_mode == H5_HEAD_SELECTED ? (int)head_rows.size() : 0;
      const int* hr = nh ? head_rows.data() : nullptr;
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
        status = h5_step_visual_head(e, rs.size(), rs.data(), rt.data(), rp.data(), rope.data(), embedding ? visual.data() : nullptr, ro.data(), head_mode, nh, hr);
      } else status = h5_step_head(e, rs.size(), rs.data(), rt.data(), rp.data(), ro.data(), head_mode, nh, hr);
      }
      if (status) { fprintf(stderr, "step failed: %d\n", status); abort(); }
      if (relax.active || relax_stats_env) {
        rstat.resize(rs.size());
        if (h5_row_stats(e, rs.size(), rstat.data())) { fprintf(stderr, "relax policy: row statistics unavailable\n"); abort(); }
      }
      const double native_s = now_s() - t_native0;
      verify_ns.fetch_add((long long)(native_s * 1e9), std::memory_order_relaxed);
      for (int s = 0; s < S; s++) if (sl[s].r && nr[s]) {
        if (sl[s].anchor < 0) sl[s].r->native_prefill_s += native_s;
        else sl[s].r->native_decode_s += native_s;
      }
      total_steps++;
      // Sampled target draws, per slot. Multi-slot sampling is non-speculative (validated at admission), so each
      // sampled slot owns either its one finishing-prefill head row or its single decode row.
      for (int s = 0; s < S; s++) {
        if (!sl[s].r || !sl[s].r->sampling.ordinary() || !nr[s]) continue;
        auto& q = sl[s];
        if (!(q.anchor >= 0 || q.done + nr[s] == (int)q.r->prompt.size())) continue;
        auto& request = *q.r;
        const bool prefill = q.anchor < 0;
        int row;
        if (head_mode == H5_HEAD_SELECTED) row = hstart[s];
        else if (S == 1) row = prefill ? (int(rs.size()) - 1) % 128 : 0;
        else row = -1;
        const int rows = prefill ? 1 : nr[s];
        const int output_row = r0[s] + (prefill ? nr[s] - 1 : 0);
        const auto config = h5::sampling::native_config(request.sampling);
        const double sampling_begin = now_s();
        const int code = row < 0 || head_rows.size() > 128 ? H5_HEAD_LOGITS_UNAVAILABLE
          : h5_sample_target(e, &config, request.sampling.seed, q.ngen,
              request.sampled_seen.data(), prefill ? nullptr : q.dr, row, rows, ro.data() + output_row);
        request.sampling_s += now_s() - sampling_begin;
        if (code) {
          request.commitment_error = "target sampling failed: " + std::to_string(code);
          finish(q, "error"); settle(q); q.r.reset(); active--; nr[s] = 0;
        }
      }
      ds.clear(); dni.clear(); dir.clear(); dip.clear(); dan.clear(); dst.clear(); dslot.clear();
      for (int s = 0; s < S; s++) {
        Sl& q = sl[s]; if (!q.r || nr[s] == 0) continue;
        auto& P = q.r->prompt;
        const bool ordinary = !q.r->use_draft();
        int ning, ipos0, an = -1, ingest_slot = s; bool fin = false;
        if (q.anchor < 0) {
          ning = nr[s]; ipos0 = q.done; q.done += nr[s];
          if (q.done == (int)P.size()) { q.len = P.size(); an = ro[r0[s] + nr[s] - 1]; total_tokens++; fin = emit(q, an); if (lookup_extra_max && !q.r->has_image()) q.lookup.append(&an, 1); h5::relax::observe(relax, q.relax_state, &an, 1); if (adjusting()) dry_push(q, an); }
        } else {
          if (relax.active && q.r->use_draft() && s != tree_s) {
            const long before_offered = relax_counters.offered, before_allowed = relax_counters.by_reason[0];
            h5::relax::apply(relax, q.relax_state, ro.data() + r0[s], rstat.data() + r0[s], q.dr, q.nd, q.nd - q.lookup_extra,
                             relax_margin, digit_table, relax_counters);
            const long off = relax_counters.offered - before_offered, ok = relax_counters.by_reason[0] - before_allowed;
            q.r->relax_offered += off; q.r->relax_vetoed += off - ok;
          }
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
            verify_rounds++; accepted_drafts += a; proposed_drafts += proposed; verifier_rows_total += nr[s];
            if (round_trace && !two && commitment.mode == h5::commit::Mode::Fixed) {
              fprintf(round_trace, "%d %d %d", nact, q.nd, a);
              for (int p = 0; p < proposals; p++) fprintf(round_trace, " %.3f", q.gaps[p]);
              fputc('\n', round_trace);
            }
            h5_accept(e, ingest_slot, a + 1);
          }
          ning = a + 1; ipos0 = q.len;
          auto target = [&](int k) { return two ? h5::selected_target(ro.data() + r0[s], 8, q.fork, choice, k) : ro[r0[s] + k]; };
          if (loop_close.active && q.r->loop_close && !two && q.lc_think) {
            // Accepted drafts 0..a-1 extend the output; if it is now periodic, the bonus becomes </think>.
            if (const int p = h5::loopclose::period(loop_close, q.out, ro.data() + r0[s], a)) {
              ro[r0[s] + a] = think_close; loop_closes++; q.r->loop_closes++;
              fprintf(stderr, "loop-close: period %d tokens at output %zu\n", p, q.out.size() + a);
            }
          }
          if (loop_close.active) for (int k = 0; k <= a; k++) {
            const int t = target(k); q.out.push_back(t);
            if (t == think_open) q.lc_think = true; else if (t == think_close) q.lc_think = false;
          }
          int emitted_drafts = 0, emitted_greedy = 0;
          std::vector<int> emitted_ids;
          const int output_start = q.r->completion_tokens;
          if (q.lookup_extra) { lookup_rounds++; lookup_proposed += q.lookup_extra; lookup_accepted += std::max(0, a - (q.nd - q.lookup_extra)); }
          if (lookup_extra_max && !q.r->has_image()) for (int k = 0; k <= a; k++) { const int t = target(k); q.lookup.append(&t, 1); }
          if (relax.active) for (int k = 0; k <= a; k++) { const int t = target(k); h5::relax::observe(relax, q.relax_state, &t, 1); }
          if (adjusting()) for (int k = 0; k <= a; k++) dry_push(q, target(k));
          for (int k = 0; k <= a && !fin; k++) {
            const int before = q.r->completion_tokens;
            total_tokens++; fin = emit(q, target(k), k);
            if (q.r->completion_tokens > before) {
              if (k < a) emitted_drafts++; else emitted_greedy++;
              if (commitment.mode != h5::commit::Mode::Fixed) emitted_ids.push_back(target(k));
            }
          }
          if (q.r->use_draft() && q.len + proposals + 1 < ctx && commitment.mode != h5::commit::Mode::Fixed) {
            const bool matched_eos = a > 0 && h5::draft_eos(q.dr[a - 1]);
            const uint64_t schedule_position = q.commitment_state.schedule_round + q.commitment_state.schedule_offset;
            const auto update = h5::commit::observe(commitment, q.commitment_state, q.nd, a, matched_eos);
            if (update.error != h5::commit::Error::None) { fprintf(stderr, "commitment accounting defect: %s\n", h5::commit::error(update.error)); abort(); }
            const auto& selection = q.commitment_selection;
            if (commitment.mode != h5::commit::Mode::Nested) q.r->commitment_rounds.push_back({{"ordinal", q.r->verify_rounds - 1}, {"proposal_count", proposals},
              {"selected_prefix", q.nd}, {"accepted_prefix", a}, {"natural_gate_rejection", update.gate_rejection},
              {"matched_eos", matched_eos}, {"selection_trigger", h5::commit::name(selection.trigger)},
              {"selection_limit", selection.limit}, {"selected_gap", selection.low_position >= 0 ? json(selection.low_gap) : json(nullptr)},
              {"next_feedback_limit", update.next_limit}, {"output_start", output_start}, {"input_position", q.len},
              {"emitted_tokens", emitted_drafts + emitted_greedy}, {"emitted_drafts", emitted_drafts},
              {"emitted_greedy_tokens", emitted_greedy}, {"greedy_kind", emitted_greedy ? (a == q.nd ? "selected-prefix-bonus" : "gate-correction") : "none"},
              {"verifier_rows", nr[s]}, {"committed_inputs", a + 1},
              {"proposal_token_ids", std::vector<int>(q.dr, q.dr + proposals)},
              {"proposal_gaps", std::vector<float>(q.gaps, q.gaps + proposals)},
              {"verified_policy_output_ids", std::vector<int>(ro.data() + r0[s], ro.data() + r0[s] + nr[s])},
              {"emitted_token_ids", emitted_ids}});
            if (commitment.mode == h5::commit::Mode::Scheduled) {
              auto& round = q.r->commitment_rounds.back();
              round["schedule_offset"] = q.commitment_state.schedule_offset;
              round["schedule_index"] = schedule_position % h5::commit::SCHEDULE_LENGTH;
              round["schedule_cycle"] = schedule_position / h5::commit::SCHEDULE_LENGTH;
            }
            // H5_COMMIT_TRACE=FILE: one JSON line per adaptive round (calibration data for the expected policy).
            static FILE* commit_trace = getenv("H5_COMMIT_TRACE") ? fopen(getenv("H5_COMMIT_TRACE"), "a") : nullptr;
            // proposals = verified proposal ids (after any lookup extension), targets = verifier policy outputs
            // (row k: the token the target emits after proposal k-1; relaxed rows already substituted).
            if (commit_trace) {
              fprintf(commit_trace, "%s\n", json{{"selected", q.nd}, {"accepted", a}, {"matched_eos", matched_eos},
                {"gaps", std::vector<float>(q.gaps, q.gaps + proposals)}, {"trigger", h5::commit::name(selection.trigger)},
                {"lookup_extra", q.lookup_extra}, {"input_position", q.len},
                {"proposals", std::vector<int>(q.dr, q.dr + q.nd)},
                {"targets", std::vector<int>(ro.data() + r0[s], ro.data() + r0[s] + nr[s])},
                {"greedy", [&] { std::vector<int> g; if (relax_stats_env) for (int k = 0; k < nr[s]; ++k) g.push_back(rstat[r0[s] + k].greedy); return g; }()},
                {"gap", [&] { std::vector<float> g; if (relax_stats_env) for (int k = 0; k < nr[s]; ++k) g.push_back(rstat[r0[s] + k].best - rstat[r0[s] + k].score); return g; }()},
                {"p1", [&] { std::vector<float> g; if (relax_stats_env) for (int k = 0; k < nr[s]; ++k) g.push_back(std::exp(rstat[r0[s] + k].best - rstat[r0[s] + k].lse)); return g; }()}}.dump().c_str()); fflush(commit_trace); }
          }
          q.len += a + 1; an = target(a);
        }
        if (!fin && q.len + 1 >= ctx) { finish(q, "length"); fin = true; }
        if (fin) { settle(q); q.r.reset(); active--; an = -1; }
        if (an >= 0) q.anchor = an;
        if (ordinary || q.len + proposals + 1 >= ctx) { q.nd = 0; continue; }
        ds.push_back(ingest_slot); dni.push_back(ning); dir.push_back(r0[s] + (ingest_slot == S ? 8 : 0)); dip.push_back(ipos0); dan.push_back(an); dst.push_back(q.len);
        if (an >= 0) dslot.push_back(s);
      }
      if (lz_draft && g_spec && !dslot.empty()) {
        // Drafter LZ: each draft block gets its request's committed generated tail and strength (0 = off).
        const int W = lz.window, nb = (int)dslot.size();
        lzd_ctx.assign((size_t)nb * W, 0); lzd_len.assign(nb, 0); lzd_alpha.assign(nb, 0.f);
        for (int b = 0; b < nb; ++b) {
          const Sl& q = sl[dslot[b]];
          if (!q.r || !q.r->lz_draft || !(q.r->lz_alpha > 0.f) || q.r->has_image()) continue;
          if (lz.scope == h5::lz::Scope::Think && !q.in_think) continue;
          const long prompt = (long)q.r->prompt.size(), ngen = (long)q.seq.size() - prompt;
          const int n = (int)std::min<long>(W, ngen);
          std::copy(q.seq.end() - n, q.seq.end(), lzd_ctx.begin() + (size_t)b * W);
          lzd_len[b] = n; lzd_alpha[b] = q.r->lz_draft_alpha >= 0.f ? q.r->lz_draft_alpha : q.r->lz_alpha;
        }
        if (h5_set_draft_lz(e, nb, lzd_ctx.data(), lzd_len.data(), lzd_alpha.data())) { fprintf(stderr, "drafter LZ unavailable\n"); abort(); }
      }
      const double t_draft0 = now_s();
      const bool two_proposal = tree2 && ds.size() == 1 && dslot.size() == 1 && active.load() == 1 && !sl[dslot[0]].r->has_image();
      int draft_status = 0;
      {
      h5trace::Range draft_range("drafter-forward");
      if (!ds.empty() && g_spec) {
        if (S == 1 && sl[0].r && sl[0].r->sampling.ordinary()) {
          const auto& request = *sl[0].r;
          const auto config = h5::sampling::native_config(request.draft_sampling);
          draft_status = h5_draft_sampled(e, ds.size(), ds.data(), dni.data(), dir.data(), dip.data(), dan.data(), dst.data(), drafts.data(),
            &config, request.sampling.seed, sl[0].ngen, request.sampled_seen.data());
        } else draft_status = two_proposal
          ? h5_draft_two(e, ds.size(), ds.data(), dni.data(), dir.data(), dip.data(), dan.data(), dst.data(), drafts.data(), alternatives.data(), forks.data(), 3)
          : h5_draft(e, ds.size(), ds.data(), dni.data(), dir.data(), dip.data(), dan.data(), dst.data(), drafts.data());
      }
      }
      if (draft_status && S == 1 && sl[0].r && sl[0].r->sampling.ordinary()) {
        auto& q = sl[0];
        q.r->commitment_error = "draft sampling failed: " + std::to_string(draft_status);
        finish(q, "error"); settle(q); q.r.reset(); active--;
        continue;
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
      if (!g_spec) dslot.clear();
      for (size_t b = 0; b < dslot.size(); b++) {
        Sl& q = sl[dslot[b]];
        memcpy(q.dr, &drafts[b * proposals], proposals * sizeof(int));
        if (cf) memcpy(q.gaps, cf + b * proposals, proposals * sizeof(float));
        q.alt_valid = two_proposal;
        if (two_proposal) { memcpy(q.alt, &alternatives[b * 7], 7 * 4); q.fork = forks[b]; }
        q.nd = proposals;
        if (commitment.mode == h5::commit::Mode::Fixed) {
          if (bspec.mode == h5::bspec::Mode::EV) {}   // whole-step selection below
          else if (cut >= 0.f) { q.nd = 0; while (q.nd < proposals && cf[b * proposals + q.nd] >= cut) q.nd++; }
          if (bspec.mode == h5::bspec::Mode::Cap) q.nd = std::min(q.nd, h5::bspec::cap_for(bspec, nact));
        } else {
          q.commitment_selection = h5::commit::select(commitment, q.commitment_state, cf ? cf + b * proposals : nullptr, proposals);
          if (q.commitment_selection.error != h5::commit::Error::None) {
            q.r->commitment_error = h5::commit::error(q.commitment_selection.error);
            finish(q, "error"); settle(q); q.r.reset(); active--; continue;
          }
          q.nd = q.commitment_selection.selected;
        }
        q.lookup_extra = 0;
        if (lookup_extra_max && !q.r->has_image()) {
          // Agreement extension: when the drafter's first `minimum` proposals equal the continuation of the longest
          // earlier occurrence of the current suffix, verify that continuation beyond them instead.
          const int agree = commitment.minimum;
          const auto hit = q.lookup.continuation(agree + lookup_extra_max);
          if (hit.length > agree && std::equal(q.dr, q.dr + agree, hit.tokens)) {
            const int extra = std::min(hit.length - agree, lookup_extra_max);
            std::copy(hit.tokens + agree, hit.tokens + agree + extra, q.dr + agree);
            q.nd = agree + extra; q.lookup_extra = extra;
          }
        }
        q.limit_drafts(q.r->draft_verify_limit);
      }
      if (bspec.mode == h5::bspec::Mode::EV && commitment.mode == h5::commit::Mode::Fixed && !dslot.empty() && cf && nact > all_below) {   // at <= H5_VERIFY_ALL_BELOW streams every proposal is verified, as in the cut policy
        std::vector<int> pick(dslot.size());
        h5::bspec::select_ev(bspec, dslot.size(), proposals, cf, 0, pick.data());
        for (size_t b = 0; b < dslot.size(); b++) if (sl[dslot[b]].r && !sl[dslot[b]].lookup_extra) {
          sl[dslot[b]].nd = pick[b]; sl[dslot[b]].limit_drafts(sl[dslot[b]].r->draft_verify_limit);
        }
      }
      busy_ns.fetch_add((long long)((now_s() - t_step0) * 1e9), std::memory_order_relaxed);
      if (round_trace && (total_steps.load() & 63) == 0) fflush(round_trace);
    }
  }
};

static json serving_config(const h5::serving::Config& c) {
  const bool prefix = !c.prefix_file.empty();
  return {{"route", prefix ? (c.fair ? "fair128-prefix" : "fifo-prefix") : (c.fair ? "fair128" : "fifo")},
    {"prefill_quantum", c.fair ? json(h5::serving::QUANTUM) : json(nullptr)},
    {"aging_s", c.fair ? json(1.) : json(nullptr)}, {"request_slots", c.request_slots},
    {"physical_slots", c.physical_slots}, {"prefix_enabled", prefix}, {"prefix_reserved", c.reserve},
    {"prefix_template_slot", c.template_slot}, {"prefix_match", "strict-token-prefix-text-only"}};
}

static std::string jd(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }
static std::string rid() { static std::atomic<long> n{0}; char b[64]; snprintf(b, sizeof b, "h5-%lx-%ld", (long)time(nullptr), n++); return b; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: h5-server MODEL DRAFTER [--host H] [--port P] [--slots S] [--ctx C] [--prefill N] [--mmproj PATH] [--vision-gpu 0|1] [--image-max-tokens N] [--draft-proposals 1..64] [--commit-policy fixed|periodic|confidence|feedback|scheduled] [--commit-min K --commit-max K --commit-confidence GAP --commit-growth N --commit-schedule-seed N --commit-calibration A,B,C]\n"); return 1; }
  const char* drafter = getenv("H5_DRAFTER");
  const bool dspark = drafter && !strcmp(drafter, "dspark-bf16");
  if (drafter && !dspark && strcmp(drafter, "dflash2")) {
    fprintf(stderr, "H5_DRAFTER must be dflash2 or dspark-bf16\n"); return 1;
  }
  const char* drafter_name = dspark ? "dspark-bf16" : "dflash2";
  json drafter_projection_precision = nullptr;
  if (dspark) {
    auto precision = h5::read_dspark_precision(getenv("H5_DSPARK_PRECISION"));
    if (const auto* error = std::get_if<h5::DsparkPrecisionError>(&precision)) {
      fprintf(stderr, "DsparkPrecisionError: %s\n", error->detail.c_str()); return 1;
    }
    drafter_projection_precision = h5::dspark_precision_name(std::get<h5::DsparkPrecision>(precision));
  } else if (getenv("H5_DSPARK_PRECISION")) {
    fprintf(stderr, "DsparkPrecisionError: H5_DSPARK_PRECISION requires the native DSpark drafter\n"); return 1;
  }
  float draft_cut = 1.f;
  int verify_all_below = 4;
  if (const char* value = getenv("H5_DRAFT_CUT")) {
    char* end = nullptr; errno = 0; draft_cut = strtof(value, &end);
    if (errno || end == value || *end || !std::isfinite(draft_cut)) {
      fprintf(stderr, "DraftSelectionError: H5_DRAFT_CUT must be a finite threshold\n"); return 1;
    }
  }
  if (const char* value = getenv("H5_VERIFY_ALL_BELOW")) {
    if (h5::commit::integer(value, verify_all_below) != h5::commit::Error::None) {
      fprintf(stderr, "DraftSelectionError: H5_VERIFY_ALL_BELOW must be a nonnegative integer\n"); return 1;
    }
  }
  const json batch_proposal_selection = {{"all_proposals_when_active_lte", verify_all_below},
    {"otherwise_prefix_threshold", draft_cut}, {"comparison", ">="},
    {"signal", dspark ? "learned-acceptance-probability" : "draft-selector-best-minus-second-raw-score"},
    {"negative_threshold_selects_all", true}};
  double draft_temperature_scale;
  std::string draft_scale_error;
  if (!h5::sampling::parse_draft_temperature_scale(getenv("H5_DRAFT_TEMPERATURE_SCALE"), draft_temperature_scale, draft_scale_error)) {
    fprintf(stderr, "%s\n", draft_scale_error.c_str()); return 1;
  }
  fprintf(stderr, "draft_temperature_scale=%.17g\n", draft_temperature_scale);
  const bool tree2 = getenv("H5_TREE2") && atoi(getenv("H5_TREE2")) != 0;
  H5HeadMode head_mode = H5_HEAD_SELECTED;
  if (const char* mode = getenv("H5_HEAD_ROWS")) {
    if (h5::parse_head_mode(mode, head_mode) != H5_HEAD_OK) {
      fprintf(stderr, "H5_HEAD_ROWS must be all or demand\n"); return 1;
    }
  }
  fprintf(stderr, "head_rows=%s\n", head_mode == H5_HEAD_ALL ? "all" : "demand");
  if (tree2 && getenv("H5_DRAFT_MARGIN") && atof(getenv("H5_DRAFT_MARGIN")) > 0.f) {
    fprintf(stderr, "H5_TREE2 requires exact argmax: H5_DRAFT_MARGIN must be 0.\n"); return 1;
  }
  std::string host = "0.0.0.0", mmproj; int port = 8000, S = 32, ctx = 8192, budget = 512;
  bool vision_gpu = false; int max_image_tokens = 1024, proposals = dspark ? h5::DSPARK_MAX_PROPOSALS : h5::DRAFT_BASE_PROPOSALS;
  h5::commit::Options commit_options; h5::commit::Config commitment;
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
    else if (k == "--commit-policy") commit_options.mode = argv[i + 1];
    else if (k == "--commit-min") commit_options.minimum = argv[i + 1];
    else if (k == "--commit-max") commit_options.maximum = argv[i + 1];
    else if (k == "--commit-confidence") commit_options.threshold = argv[i + 1];
    else if (k == "--commit-growth") commit_options.growth = argv[i + 1];
    else if (k == "--commit-schedule-seed") commit_options.schedule_seed = argv[i + 1];
    else if (k == "--commit-calibration") commit_options.calibration = argv[i + 1];
    else { fprintf(stderr, "unknown server option: %s\n", k.c_str()); return 1; }
  }
  g_spec = !(getenv("H5_NO_SPEC") && !strcmp(getenv("H5_NO_SPEC"), "1"));
  h5::bspec::Policy bspec;
  if (const auto be = h5::bspec::configure(getenv("H5_SPEC_POLICY"), getenv("H5_SPEC_CAP"), getenv("H5_SPEC_PCAL"), getenv("H5_SPEC_COST"), bspec); be != h5::bspec::Error::None) {
    fprintf(stderr, "%s\n", h5::bspec::error(be)); return 1;
  }
  h5::serving::Config serving;
  const auto serving_error = h5::serving::configure(getenv("H5_SERVING_FAIR"), getenv("H5_SERVING_RESERVE"),
    getenv("H5_SERVING_PREFIX"), getenv("H5_SERVING_CHECK"), S, tree2, serving);
  if (serving_error != h5::serving::Error::None) { fprintf(stderr, "serving: %s\n", h5::serving::error(serving_error)); return 1; }
  if (!serving.prefix_file.empty() && getenv("H5_REPLAY_WRITE")) {
    fprintf(stderr, "prefix serving cannot record a replay without the prebuilt donor state\n"); return 1;
  }
  const auto draft_cfg = dspark ? h5::dspark_config(proposals, serving.request_slots, tree2, ctx, budget)
                               : h5::draft_config(proposals, serving.request_slots, tree2, ctx, budget);
  if (draft_cfg.error != h5::DraftError::None) { fprintf(stderr, "%s\n", h5::draft_error(draft_cfg.error)); return 1; }
  const bool checkpoint_length_extrapolation = !dspark && proposals != h5::DRAFT_BASE_PROPOSALS;
  if (proposals != 7 && !mmproj.empty()) { fprintf(stderr, "%s supports text only\n", dspark ? "DSpark" : "experimental long draft"); return 1; }
  if (dspark && (commit_options.mode == "nested" ||
      (getenv("H5_DRAFT_NEST") && h5::parse_proposals(getenv("H5_DRAFT_NEST")) != 0))) {
    fprintf(stderr, "DSpark does not support nested commitment or H5_DRAFT_NEST (except 0)\n"); return 1;
  }
  const auto commit_error = h5::commit::configure(commit_options, proposals, commitment);
  if (commit_error != h5::commit::Error::None) { fprintf(stderr, "commitment: %s\n", h5::commit::error(commit_error)); return 1; }
  if (commitment.mode != h5::commit::Mode::Fixed && (serving.request_slots != 1 || tree2 || !mmproj.empty())) {
    fprintf(stderr, "adaptive commitment requires one text-only slot, no H5_TREE2/mmproj\n"); return 1;
  }
  fprintf(stderr, "commitment %s; batch proposal selection %s\n", commitment_config(commitment, dspark).dump().c_str(), batch_proposal_selection.dump().c_str());
  // H5_LOOKUP_EXTRA=E: prompt-lookup agreement extension, at most E proposals appended after the nested short prefix.
  int lookup_extra_max = 0;
  if (const char* le = getenv("H5_LOOKUP_EXTRA")) {
    lookup_extra_max = h5::parse_proposals(le);
    if (lookup_extra_max != 0 && (commitment.mode != h5::commit::Mode::Nested || lookup_extra_max < 1 || lookup_extra_max > proposals - commitment.minimum)) {
      fprintf(stderr, "H5_LOOKUP_EXTRA must be 0, or 1..(proposals - commit-min) with --commit-policy nested\n"); return 1;
    }
  }
  fprintf(stderr, "lookup_extra_max=%d\n", lookup_extra_max);
  h5::relax::Policy relax;
  if (const std::string err = h5::relax::parse(getenv("H5_RELAX_POLICY"), relax); !err.empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  float relax_margin = 0.f;
  if (relax.active) {
    if (!getenv("H5_DRAFT_MARGIN") || !h5::parse_draft_margin(getenv("H5_DRAFT_MARGIN"), relax_margin) || !(relax_margin > 0.f)) {
      fprintf(stderr, "H5_RELAX_POLICY requires H5_DRAFT_MARGIN > 0\n"); return 1;
    }
    if (tree2) { fprintf(stderr, "H5_RELAX_POLICY excludes H5_TREE2\n"); return 1; }
    const int need = relax.stats_mode();
    const char* have = getenv("H5_RELAX_STATS");
    if (have && atoi(have) < need) { fprintf(stderr, "H5_RELAX_POLICY needs H5_RELAX_STATS >= %d\n", need); return 1; }
    if (!have) setenv("H5_RELAX_STATS", need == 2 ? "2" : "1", 1);
  }
  h5::dry::Config dry;
  if (const std::string err = h5::dry::parse(getenv("H5_DRY"), dry); !err.empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  if (dry.active) {
    if (tree2) { fprintf(stderr, "H5_DRY excludes H5_TREE2\n"); return 1; }
    if (!getenv("H5_LOGIT_ADJ")) setenv("H5_LOGIT_ADJ", "16", 1);
    if (atoi(getenv("H5_LOGIT_ADJ")) < 1) { fprintf(stderr, "H5_DRY needs H5_LOGIT_ADJ >= 1\n"); return 1; }
    if (!h5::dry::parse_scope(getenv("H5_DRY_SCOPE"), dry.scope)) { fprintf(stderr, "H5_DRY needs H5_DRY_SCOPE=all|output|think\n"); return 1; }
  }
  bool lz_on = false; h5::lz::Config lz{};
  unsetenv("H5_DRAFT_LZ");   // engine-side switch owned here: set only from H5_LZ + H5_LZ_DRAFT=on
  if (const char* lzt = getenv("H5_LZ"); lzt && strcmp(lzt, "none")) {
    if (const std::string err = h5::lz::parse(lzt, getenv("H5_LZ_SCOPE"), lz); !err.empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (tree2) { fprintf(stderr, "H5_LZ excludes H5_TREE2\n"); return 1; }
    lz_on = true;
    const char* ld = getenv("H5_LZ_DRAFT");
    if (!ld || (strcmp(ld, "on") && strcmp(ld, "off"))) { fprintf(stderr, "H5_LZ needs H5_LZ_DRAFT=on|off (drafter-side penalty)\n"); return 1; }
    if (!strcmp(ld, "on")) {
      if (lz.window > 1024) { fprintf(stderr, "H5_LZ_DRAFT=on supports WINDOW <= 1024\n"); return 1; }
      setenv("H5_DRAFT_LZ", (std::to_string(lz.window) + ":" + std::to_string(lz.buffer) + ":" + h5::lz::name(lz.form)).c_str(), 1);
    }
    // Engine slots per row: DRY's share (H5_LOGIT_ADJ as set for DRY, else 16) plus one per window token.
    const int dry_share = dry.active ? atoi(getenv("H5_LOGIT_ADJ")) : 0;
    setenv("H5_LOGIT_ADJ", std::to_string(dry_share + lz.window).c_str(), 1);
  }
  fprintf(stderr, "lz_draft=%s\n", getenv("H5_DRAFT_LZ") ? getenv("H5_DRAFT_LZ") : "off");
  fprintf(stderr, "dry=%s lz=%s scope=%s logit_adj=%s\n", getenv("H5_DRY") ? getenv("H5_DRY") : "none", lz_on ? getenv("H5_LZ") : "none",
          lz_on ? (lz.scope == h5::lz::Scope::Think ? "think" : "output") : "-", getenv("H5_LOGIT_ADJ") ? getenv("H5_LOGIT_ADJ") : "0");
  fprintf(stderr, "relax_policy=%s stats=%s\n", getenv("H5_RELAX_POLICY") ? getenv("H5_RELAX_POLICY") : "none", getenv("H5_RELAX_STATS") ? getenv("H5_RELAX_STATS") : "0");
  fprintf(stderr, "drafter=%s draft proposals=%d query_rows=%d verifier_rows=%d (anchor+proposals; last output is greedy bonus) checkpoint_length_extrapolation=%s\n", drafter_name, proposals, draft_cfg.query_rows, draft_cfg.verifier_rows, checkpoint_length_extrapolation ? "true" : "false");
  llama_backend_init(); ggml_backend_load_all();
  Tok tk(argv[1], !mmproj.empty()); if (!tk.m) { fprintf(stderr, "tokenizer load failed\n"); return 1; }
  std::string model_id = argv[1];
  const auto slash = model_id.find_last_of("/\\");
  if (slash != std::string::npos) model_id.erase(0, slash + 1);
  if (model_id.size() >= 5 && model_id.compare(model_id.size() - 5, 5, ".gguf") == 0) model_id.resize(model_id.size() - 5);
  if (model_id.empty()) { fprintf(stderr, "checkpoint filename has no model name\n"); return 1; }
  fprintf(stderr, "model_id=%s\n", model_id.c_str());
  h5::Vision vision; std::string vision_error;
  if (!mmproj.empty() && !vision.init(mmproj.c_str(), tk.m, vision_gpu, max_image_tokens, vision_error)) {
    fprintf(stderr, "vision: %s\n", vision_error.c_str()); return 1;
  }
  Scheduler sch; sch.tk = &tk; sch.S = serving.request_slots; sch.ctx = ctx; sch.budget = budget;
  sch.cut_many = draft_cut; sch.all_below = verify_all_below;
  sch.bspec = bspec;
  if (const char* rt = getenv("H5_ROUND_TRACE")) { sch.round_trace = fopen(rt, "a"); if (!sch.round_trace) { fprintf(stderr, "cannot open H5_ROUND_TRACE\n"); return 1; } setvbuf(sch.round_trace, nullptr, _IOFBF, 1 << 20); }
  if (!g_spec && (!serving.prefix_file.empty() || tree2 || commitment.mode != h5::commit::Mode::Fixed)) { fprintf(stderr, "H5_NO_SPEC excludes prefix serving, H5_TREE2 and adaptive commitment\n"); return 1; }
  fprintf(stderr, "speculation=%s policy=%s\n", g_spec ? "on" : "off", bspec.mode == h5::bspec::Mode::Cut ? "cut" : bspec.mode == h5::bspec::Mode::Cap ? "cap" : "ev");
  sch.relax = relax; sch.relax_margin = relax_margin;
  sch.dry = dry; sch.lz_on = lz_on; sch.lz = lz; sch.lz_draft = getenv("H5_DRAFT_LZ") != nullptr;
  sch.adj_k = sch.adjusting() ? atoi(getenv("H5_LOGIT_ADJ")) : 0;
  sch.dry_k = dry.active ? sch.adj_k - (lz_on ? lz.window : 0) : 0;
  sch.dry_tok.assign(std::max(1, sch.dry_k), -1); sch.dry_val.assign(std::max(1, sch.dry_k), 0.f);
  if (const std::string err = h5::loopclose::parse(getenv("H5_LOOP_CLOSE"), sch.loop_close); !err.empty()) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  if (sch.loop_close.active && tree2) { fprintf(stderr, "H5_LOOP_CLOSE excludes H5_TREE2\n"); return 1; }
  if ((dry.active && dry.scope == h5::dry::Scope::Think) || (lz_on && lz.scope == h5::lz::Scope::Think) || sch.loop_close.active) {
    const auto o = tk.enc("<think>"), c = tk.enc("</think>");
    if (o.size() != 1 || c.size() != 1) { fprintf(stderr, "think scope / H5_LOOP_CLOSE: <think>/</think> are not single tokens\n"); return 1; }
    sch.think_open = o[0]; sch.think_close = c[0];
    fprintf(stderr, "think tokens %d %d\n", sch.think_open, sch.think_close);
  }
  if (const char* mk = getenv("H5_MARKERS"); mk && strcmp(mk, "none")) {
    // H5_MARKERS=LAMBDA:FILE: FILE is JSON {"surfaces": [...]}; each surface must be exactly one token.
    const char* colon = strchr(mk, ':');
    char* end = nullptr; const float lambda = colon ? strtof(mk, &end) : -1.f;
    if (!colon || end != colon || !(lambda >= 0.f && lambda <= 8.f)) { fprintf(stderr, "H5_MARKERS must be LAMBDA:FILE with LAMBDA in [0,8]\n"); return 1; }
    if (tree2) { fprintf(stderr, "H5_MARKERS excludes H5_TREE2\n"); return 1; }
    std::ifstream f(colon + 1);
    if (!f) { fprintf(stderr, "H5_MARKERS: cannot read %s\n", colon + 1); return 1; }
    json spec; try { spec = json::parse(f); } catch (const json::exception& ex) { fprintf(stderr, "H5_MARKERS: %s\n", ex.what()); return 1; }
    if (!spec.contains("surfaces") || !spec["surfaces"].is_array()) { fprintf(stderr, "H5_MARKERS: file needs a \"surfaces\" array\n"); return 1; }
    std::vector<std::string> split;
    for (const auto& x : spec["surfaces"]) {
      if (!x.is_string()) { fprintf(stderr, "H5_MARKERS: surfaces must be strings\n"); return 1; }
      const auto t = tk.enc(x.get<std::string>());
      if (t.size() == 1) { if (std::find(sch.markers.begin(), sch.markers.end(), t[0]) == sch.markers.end()) sch.markers.push_back(t[0]); }
      else split.push_back(x.get<std::string>());
    }
    if (sch.markers.empty()) { fprintf(stderr, "H5_MARKERS: no surface is a single token\n"); return 1; }
    fprintf(stderr, "markers: %zu tokens, lambda %.3g; not single tokens (unpenalized):", sch.markers.size(), lambda);
    for (auto& x : split) fprintf(stderr, " [%s]", x.c_str());
    fputc('\n', stderr);
    sch.marker_lambda = lambda;
    if (sch.think_open < 0) {
      const auto o = tk.enc("<think>"), c = tk.enc("</think>");
      if (o.size() != 1 || c.size() != 1) { fprintf(stderr, "H5_MARKERS: <think>/</think> are not single tokens\n"); return 1; }
      sch.think_open = o[0]; sch.think_close = c[0];
    }
    if (!sch.dry.active && !sch.lz_on) sch.adj_k = 0;
    sch.adj_k += (int)sch.markers.size();
    setenv("H5_LOGIT_ADJ", std::to_string(sch.adj_k).c_str(), 1);
  }
  if (const char* ls = getenv("H5_LOOKUP_SCOPE")) {
    if (!strcmp(ls, "prompt")) sch.lookup_prompt_only = true; else if (strcmp(ls, "all")) { fprintf(stderr, "H5_LOOKUP_SCOPE must be all or prompt\n"); return 1; }
  }
  if (const char* lm = getenv("H5_LOOKUP_MIN")) { sch.lookup_min = h5::parse_proposals(lm); if (sch.lookup_min < 0 || sch.lookup_min > 24) { fprintf(stderr, "H5_LOOKUP_MIN must be 0..24\n"); return 1; } }
  fprintf(stderr, "lookup scope=%s min=%d\n", sch.lookup_prompt_only ? "prompt" : "all", sch.lookup_min);
  sch.relax_stats_env = getenv("H5_RELAX_STATS") && strcmp(getenv("H5_RELAX_STATS"), "0");
  if (relax.digit || relax.special) {
    sch.digit_table.assign(h5::VOCAB, 0);
    for (int t = 0; t < h5::VOCAB && t < llama_vocab_n_tokens(tk.v); ++t) {
      const std::string p = tk.piece(t);   // bit 0: any digit (MLX digit-strict relaxed thinking); bit 1: control token
      sch.digit_table[t] = (std::any_of(p.begin(), p.end(), [](char c) { return c >= '0' && c <= '9'; }) ? 1 : 0) |
                           (llama_vocab_is_control(tk.v, t) ? 2 : 0);
    }
  }
  sch.tree2 = tree2; sch.lookup_extra_max = lookup_extra_max; sch.proposals = proposals; sch.commitment = commitment; sch.serving = serving; sch.head_mode = head_mode;
  fprintf(stderr, "serving=%s physical_slots=%d request_slots=%d tree_scratch=%d prefix_template=%d\n",
    serving.fair ? "fair128" : "fifo", serving.physical_slots, sch.S, tree2 ? sch.S : -1, serving.template_slot);
  // Scratch slot is private to the scheduler thread and never exposed/admitted.
  sch.e = h5_create_draft(argv[1], serving.physical_slots, ctx, draft_cfg.engine_rows, proposals);
  if (!sch.e || (g_spec && h5_draft_load(sch.e, argv[2]))) return 1;
  h5_capture(sch.e, g_spec ? 1 : 0);
  h5_pretune(sch.e);
  if (!serving.prefix_file.empty() && !sch.build_prefix()) { fprintf(stderr, "prefix build failed\n"); return 1; }
  if (serving.check && !sch.check_prefix()) { fprintf(stderr, "prefix ownership/reject/reset diagnostic failed\n"); return 1; }
  std::thread worker([&] { sch.run(); });
  const double t_start = now_s();

  httplib::Server svr;
  svr.set_read_timeout(600); svr.set_write_timeout(600);
  svr.new_task_queue = [] { return new httplib::ThreadPool(256); };
  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(jd(json{{"status", "ok"}, {"context_length", ctx}, {"sampling", {{"route", g_spec ? (dspark ? "greedy-speculative-target-verification" : "noise-coupled-speculative-target-sampling") : "ordinary-gpu-target-sampling"}, {"request_slots", sch.S}, {"speculative_sampling_slots", dspark ? 0 : 1}, {"penalty_history", "generated-tokens"}, {"noise", "seed-position-token-gumbel"}, {"draft_temperature_scale", draft_temperature_scale}}}, {"head_rows", head_mode == H5_HEAD_ALL ? "all" : "demand"}, {"serving", serving_config(serving)}, {"vision", vision.enabled()}, {"vision_language_backend", "halo-native"}, {"drafter", drafter_name}, {"drafter_projection_precision", drafter_projection_precision}, {"draft_proposals", proposals}, {"draft_query_rows", draft_cfg.query_rows}, {"verifier_rows", draft_cfg.verifier_rows}, {"checkpoint_length_extrapolation", checkpoint_length_extrapolation}, {"bonus_policy", "greedy"}, {"batch_proposal_selection", batch_proposal_selection}, {"commitment_policy", commitment_config(commitment, dspark)}}), "application/json");
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
  svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    json j = {{"object", "list"}, {"data", json::array({{{"id", model_id}, {"object", "model"}, {"owned_by", "halo-5090"}}})}};
    res.set_content(jd(j), "application/json");
  });
  svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
    json j = {{"serving", serving_config(serving)}, {"batch_proposal_selection", batch_proposal_selection}, {"drafter", drafter_name}, {"drafter_projection_precision", drafter_projection_precision}, {"draft_proposals", proposals}, {"draft_query_rows", draft_cfg.query_rows}, {"verifier_rows_per_full_round", draft_cfg.verifier_rows}, {"checkpoint_length_extrapolation", checkpoint_length_extrapolation}, {"uptime_s", now_s() - t_start}, {"requests", sch.total_req.load()}, {"active", sch.active.load()},
              {"generated_tokens", sch.total_tokens.load()}, {"steps", sch.total_steps.load()}, {"busy_s", sch.busy_ns.load() * 1e-9},
              {"verify_s", sch.verify_ns.load() * 1e-9}, {"draft_s", sch.draft_ns.load() * 1e-9},
              {"verify_rounds", sch.verify_rounds.load()}, {"accepted_drafts", sch.accepted_drafts.load()},
              {"proposed_drafts", sch.proposed_drafts.load()}, {"verifier_rows", sch.verifier_rows_total.load()}, {"speculation", g_spec},
              {"tree2_enabled", sch.tree2}, {"tree_rounds", sch.tree_rounds.load()}, {"tree_rescues", sch.tree_rescues.load()},
              {"tree_fallback_rounds", sch.tree_fallback_rounds.load()},
              {"clone_s", sch.clone_ns.load() * 1e-9},
              {"lookup_extra_max", sch.lookup_extra_max}, {"lookup_rounds", sch.lookup_rounds.load()},
              {"lookup_proposed", sch.lookup_proposed.load()}, {"lookup_accepted", sch.lookup_accepted.load()},
              {"loop_closes", sch.loop_closes.load()}, {"adjust_s", sch.adjust_ns.load() * 1e-9},
              {"relax_policy", getenv("H5_RELAX_POLICY") ? getenv("H5_RELAX_POLICY") : "none"},
              {"relax_offered", sch.relax_counters.offered}, {"relax_by_reason", [&] { json r; for (int i = 0; i < (int)h5::relax::Reason::Count; ++i) r[h5::relax::name((h5::relax::Reason)i)] = sch.relax_counters.by_reason[i]; return r; }()},
              {"serving_policy", serving.fair ? "fair128" : "fifo"}, {"slots", sch.S}, {"physical_slots", serving.physical_slots},
              {"prefix_template_slot", serving.template_slot}, {"prefix_tokens", sch.prefix.size()},
              {"prefix_build_s", sch.prefix_build_s}, {"prefix_checked", sch.prefix_checked},
              {"prefix_target_changes", sch.prefix_target_changes}, {"prefix_draft_changes", sch.prefix_draft_changes},
              {"prefix_hits", sch.prefix_hits.load()}, {"prefix_misses", sch.prefix_misses.load()},
              {"prefill_rows", sch.prefill_rows.load()}, {"cancelled", sch.cancelled.load()},
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
        if (think) r->text = "<think>\n";
        if (!h5::chat_prompt(body.value("messages", json::array()), think, mtmd_default_marker(), prompt, images, error)) {
          res.status = 400; res.set_content(jd(json{{"error", {{"message", error}}}}), "application/json"); return;
        }
      } else {
        const auto& p = body.at("prompt");
        prompt = p.is_array() && !p.empty() ? p[0].get<std::string>() : p.get<std::string>();
      }
      if (!images.empty()) {
        if (commitment.mode != h5::commit::Mode::Fixed) {
          res.status = 400; res.set_content("{\"error\":{\"message\":\"adaptive commitment supports text only\"}}", "application/json"); return;
        }
        if (!vision.prepare(prompt, images, ctx - 17, r->visual, error)) {
          res.status = 400; res.set_content(jd(json{{"error", {{"message", error}}}}), "application/json"); return;
        }
        r->prompt = r->visual.tokens;
      } else r->prompt = tk.enc(prompt);
    } catch (const json::exception& ex) {
      res.status = 400; res.set_content(jd(json{{"error", {{"message", std::string("invalid request: ") + ex.what()}}}}), "application/json"); return;
    }
    const int remaining = ctx - (int)r->prompt.size();
    if (r->prompt.empty() || remaining <= 0) { res.status = 400; res.set_content(jd(json({{"error", {{"message", "prompt exceeds context " + std::to_string(ctx)}}}})), "application/json"); return; }
    r->max_new = remaining;
    for (const char* key : {"max_tokens", "max_completion_tokens"}) if (body.contains(key)) {
      const auto& limit = body[key];
      if (!limit.is_number_integer() || limit.get<double>() <= 0 || limit.get<double>() > INT32_MAX) {
        res.status = 400; res.set_content(jd(json{{"error", {{"message", std::string(key) + " must be a positive integer"}}}}), "application/json"); return;
      }
      r->max_new = std::min(remaining, limit.get<int>());
    }
    if (!h5::sampling::parse(body, r->sampling, error) || (r->sampling.ordinary() && (tree2 || !sch.prefix.empty()))) {
      if (error.empty()) error = "sampling requires serving without tree/prefix";
      res.status = 400; res.set_content(jd(json{{"error", {{"message", error}}}}), "application/json"); return;
    }
    if (!h5::sampling::parse_draft(body, r->sampling, proposals, r->draft_sampling, r->draft_verify_limit, error, draft_temperature_scale)) {
      res.status = 400; res.set_content(jd(json{{"error", {{"message", error}}}}), "application/json"); return;
    }
    if (dspark && r->sampling.ordinary() && r->use_draft()) {
      res.status = 400; res.set_content("{\"error\":{\"type\":\"invalid_request_error\",\"code\":\"dspark_sampling_unsupported\",\"message\":\"DSpark drafting is greedy-only; sampling requires draft_verify_limit 0 or H5_NO_SPEC=1\"}}", "application/json"); return;
    }
    // Speculative sampling (coupled drafter draws) is single-slot; with more slots a sampled request must not draft.
    if (r->sampling.ordinary() && sch.S != 1 && r->use_draft()) {
      res.status = 400; res.set_content("{\"error\":{\"message\":\"multi-slot sampling requires draft_verify_limit 0 or H5_NO_SPEC=1\"}}", "application/json"); return;
    }
    if (r->sampling.ordinary()) r->sampled_seen.assign(h5::VOCAB, 0);
    r->lz_alpha = sch.lz_on ? sch.lz.alpha : 0.f;
    r->lz_draft = sch.lz_draft;
    if (body.contains("lz_draft")) {
      const auto& v = body["lz_draft"];
      if (!sch.lz_draft || !v.is_boolean()) {
        res.status = 400;
        res.set_content(jd(json{{"error", {{"message", sch.lz_draft ? "lz_draft must be a boolean" : "lz_draft needs the server started with H5_LZ_DRAFT=on"}}}}), "application/json");
        return;
      }
      r->lz_draft = v.get<bool>();
    }
    if (body.contains("lz_draft_alpha")) {
      const auto& v = body["lz_draft_alpha"];
      const double a = v.is_number() ? v.get<double>() : -1.0;
      if (!sch.lz_draft || !(a >= 0.0 && a <= 8.0)) {
        res.status = 400;
        res.set_content(jd(json{{"error", {{"message", sch.lz_draft ? "lz_draft_alpha must be a number in [0,8]" : "lz_draft_alpha needs the server started with H5_LZ_DRAFT=on"}}}}), "application/json");
        return;
      }
      r->lz_draft_alpha = (float)a;
    }
    if (body.contains("lz_penalty")) {
      const auto& v = body["lz_penalty"];
      const double a = v.is_number() ? v.get<double>() : -1.0;
      if (!sch.lz_on || !(a >= 0.0 && a <= 4.0)) {
        res.status = 400;
        res.set_content(jd(json{{"error", {{"message", sch.lz_on ? "lz_penalty must be a number in [0,4]" : "lz_penalty needs the server started with H5_LZ"}}}}), "application/json");
        return;
      }
      r->lz_alpha = (float)a;
    }
    r->marker_lambda = sch.marker_lambda;
    if (body.contains("marker_penalty")) {
      const auto& v = body["marker_penalty"];
      const double a = v.is_number() ? v.get<double>() : -1.0;
      if (sch.markers.empty() || !(a >= 0.0 && a <= 8.0)) {
        res.status = 400;
        res.set_content(jd(json{{"error", {{"message", sch.markers.empty() ? "marker_penalty needs the server started with H5_MARKERS" : "marker_penalty must be a number in [0,8]"}}}}), "application/json");
        return;
      }
      r->marker_lambda = (float)a;
    }
    r->loop_close = sch.loop_close.active;
    if (body.contains("loop_close")) {
      const auto& v = body["loop_close"];
      if (!sch.loop_close.active || !v.is_boolean()) {
        res.status = 400;
        res.set_content(jd(json{{"error", {{"message", sch.loop_close.active ? "loop_close must be a boolean" : "loop_close needs the server started with H5_LOOP_CLOSE"}}}}), "application/json");
        return;
      }
      r->loop_close = v.get<bool>();
    }
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
      r->cv.wait(l, [&] { return r->done && r->settled; });
      if (r->finish == "error") {
        res.status = 500;
        res.set_content(jd(json{{"error", {{"type", "server_error"}, {"code", "inference_error"}, {"message", r->commitment_error}}}}), "application/json");
        return;
      }
      json choice = chat ? json{{"index", 0}, {"message", {{"role", "assistant"}, {"content", r->text}}}, {"finish_reason", r->finish}}
                         : json{{"index", 0}, {"text", r->text}, {"finish_reason", r->finish}};
      json j = {{"id", id}, {"object", obj}, {"created", created}, {"model", model_id}, {"choices", json::array({choice})}, {"usage", usage()},
                {"timings", {{"image_tokens", r->visual.image_tokens}, {"image_encode_s", r->visual.encode_s}, {"native_prefill_s", r->native_prefill_s},
                             {"vision_speculation", r->has_image() && r->vision_speculation},
                             {"drafter", drafter_name}, {"drafter_projection_precision", drafter_projection_precision}, {"draft_proposals", proposals}, {"draft_query_rows", draft_cfg.query_rows}, {"verifier_rows_per_full_round", draft_cfg.verifier_rows},
                             {"checkpoint_length_extrapolation", checkpoint_length_extrapolation}, {"bonus_policy", r->sampling.ordinary() ? "target-sampling" : "greedy"},
                             {"sampling_seed", r->sampling.ordinary() ? json(r->sampling.seed) : json(nullptr)},
                             {"draft_verify_limit", r->draft_verify_limit},
                             {"draft_sampling", {{"temperature", r->draft_sampling.temperature}, {"top_p", r->draft_sampling.top_p},
                               {"top_k", r->draft_sampling.top_k}, {"min_p", r->draft_sampling.min_p},
                               {"presence_penalty", r->draft_sampling.presence_penalty}, {"repetition_penalty", r->draft_sampling.repetition_penalty}}},
                             {"batch_proposal_selection", batch_proposal_selection},
                             {"commitment", {{"config", commitment_config(commitment, dspark)}, {"error", r->commitment_error.empty() ? json(nullptr) : json(r->commitment_error)}, {"rounds", r->commitment_rounds}}},
                             {"verify_rounds", r->verify_rounds}, {"verifier_rows", r->verifier_rows},
                             {"prefix_histogram", r->prefix_histogram}, {"position_proposed", r->position_proposed},
                             {"position_reached", r->position_reached}, {"position_accepted", r->position_accepted}, {"position_emitted", r->position_emitted},
                             {"position_reject_target_eos", r->position_reject_target_eos}, {"position_reject_proposed_eos", r->position_reject_proposed_eos},
                             {"position_reject_other", r->position_reject_other},
                             {"proposed_drafts", r->proposed_drafts}, {"accepted_drafts", r->accepted_drafts}, {"relax_offered", r->relax_offered}, {"relax_vetoed", r->relax_vetoed}, {"loop_closes", r->loop_closes}, {"loop_close", r->loop_close}, {"lz_penalty", r->lz_alpha}, {"lz_draft", r->lz_draft}, {"lz_draft_alpha", r->lz_draft_alpha >= 0.f ? r->lz_draft_alpha : r->lz_alpha}, {"marker_penalty", r->marker_lambda},
                             {"verify_s", r->native_decode_s}, {"draft_s", r->draft_s}, {"sampling_s", r->sampling_s},
                             {"tree_rounds", r->tree_rounds}, {"tree_rescues", r->tree_rescues}, {"clone_s", r->clone_s},
                             {"tree_fallback_rounds", r->tree_fallback_rounds},
                             {"ttft_s", r->t_first ? r->t_first - r->t_submit : 0.}, {"total_s", r->t_done - r->t_submit},
                             {"decode_tok_per_s", r->completion_tokens > 1 && r->t_done > r->t_first ? (r->completion_tokens - 1) / (r->t_done - r->t_first) : 0.}}}};
      res.set_content(jd(j), "application/json");
      return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream", [r, id, created, chat, usage_in_stream, model_id](size_t, httplib::DataSink& sink) {
      auto chunk = [&](const json& delta_or_text, const char* fin) {
        json choice = chat ? json{{"index", 0}, {"delta", delta_or_text}, {"finish_reason", fin ? json(fin) : json(nullptr)}}
                           : json{{"index", 0}, {"text", delta_or_text}, {"finish_reason", fin ? json(fin) : json(nullptr)}};
        json j = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created}, {"model", model_id}, {"choices", json::array({choice})}};
        std::string s = "data: " + jd(j) + "\n\n";
        return sink.write(s.data(), s.size());
      };
      if (chat && !chunk(json{{"role", "assistant"}, {"content", ""}}, nullptr)) { r->cancel = true; return false; }
      for (;;) {
        std::string piece; bool done; std::string fin;
        {
          std::unique_lock<std::mutex> l(r->mu);
          r->cv.wait_for(l, std::chrono::milliseconds(200), [&] { return r->settled || r->text.size() > r->sent; });
          piece = r->text.substr(r->sent); r->sent = r->text.size(); done = r->done && r->settled; fin = r->finish;
        }
        if (!piece.empty() && !chunk(chat ? json{{"content", piece}} : json(piece), nullptr)) { r->cancel = true; return false; }
        if (done) {
          if (fin == "error") {
            const std::string error = "data: " + jd(json{{"error", {{"type", "server_error"}, {"code", "inference_error"}, {"message", r->commitment_error}}}}) + "\n\n";
            sink.write(error.data(), error.size()); sink.done(); return true;
          }
          chunk(chat ? json::object() : json(""), fin.c_str());
          if (usage_in_stream) {
            json j = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created}, {"model", model_id}, {"choices", json::array()},
                      {"usage", {{"prompt_tokens", (int)r->prompt.size()}, {"completion_tokens", r->completion_tokens}, {"total_tokens", (int)r->prompt.size() + r->completion_tokens}}}};
            std::string s = "data: " + jd(j) + "\n\n"; sink.write(s.data(), s.size());
          }
          std::string end = "data: [DONE]\n\n"; sink.write(end.data(), end.size());
          sink.done();
          return true;
        }
        if (!sink.is_writable()) { r->cancel = true; return false; }
      }
    }, [r](bool) { std::lock_guard<std::mutex> l(r->mu); if (!r->done) r->cancel = true; });
  };
  svr.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, true); });
  svr.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, false); });
  fprintf(stderr, "h5-server listening on %s:%d (%d slots, ctx %d)\n", host.c_str(), port, sch.S, ctx);
  svr.listen(host, port);
  return 0;
}
