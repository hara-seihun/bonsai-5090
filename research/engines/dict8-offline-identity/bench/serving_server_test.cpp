#define main h5_http_main_unused
#include "../src/server.cpp"
#undef main
#include <cassert>
#include <sys/wait.h>
#include <unistd.h>

struct NativeSlot {
  std::vector<int> base, pending, draft_history;
  int keep = 0, commit = 0;
};
struct MockServing {
  std::vector<NativeSlot> slots;
  std::vector<int> captured;
  std::vector<float> confidence;
  int proposals = 7, donor = -1, copies = 0, visual_steps = 0, resets = 0;
  int no_head_steps = 0, partial_head_steps = 0;
  bool cancel_in_step = false;
  std::shared_ptr<Request> cancel_request;
} mock;
static int next_token(const std::vector<int>& history) {
  int hash = 0; for (int t : history) hash = (hash * 3 + t) % 101;
  return 10 + hash % 20;
}
extern "C" {
void llama_log_set(ggml_log_callback, void*) {}
llama_model_params llama_model_default_params() { return {}; }
llama_model* llama_model_load_from_file(const char*, llama_model_params) { return reinterpret_cast<llama_model*>(1); }
const llama_vocab* llama_model_get_vocab(const llama_model*) { return reinterpret_cast<const llama_vocab*>(1); }
int32_t llama_tokenize(const llama_vocab*, const char* text, int32_t len, llama_token* out, int32_t cap, bool, bool) {
  if (cap < len) return -len;
  for (int i = 0; i < len; ++i) out[i] = (unsigned char)text[i] % 9 + 1;
  return len;
}
int32_t llama_token_to_piece(const llama_vocab*, llama_token token, char* out, int32_t, int32_t, bool) {
  out[0] = 'a' + token % 26; return 1;
}
void h5_reset(void*, int s) { mock.slots.at(s) = {}; mock.resets++; }
void h5_keep(void*, int s, int keep) { mock.slots.at(s).keep = keep; }
void h5_accept(void*, int s, int n) {
  auto& q = mock.slots.at(s); assert(n > 0 && n <= (int)q.pending.size()); q.commit = n;
}
int h5_copy_prefix(void*, int src, int dst, int npos) {
  if (src < 0 || src >= (int)mock.slots.size()) return -1;
  const auto& from = mock.slots[src];
  auto e = h5::serving::copy_guard(src, dst, npos, mock.slots.size(), 4096, from.pending.size(), from.commit);
  if (e != h5::serving::CopyError::None) return (int)e;
  assert(npos == (int)from.base.size() && from.base == from.draft_history);
  mock.slots[dst] = from; mock.slots[dst].keep = mock.slots[dst].commit = 0; mock.copies++; return 0;
}
int h5_step(void*, int rows, const int* slots, const int* tokens, const int* positions, int* out) {
  mock.captured.assign(tokens, tokens + rows);
  int row = 0;
  while (row < rows) {
    int s = slots[row], end = row + 1; while (end < rows && slots[end] == s) ++end;
    auto& q = mock.slots.at(s);
    if (!q.pending.empty()) {
      assert(q.commit > 0);
      q.base.insert(q.base.end(), q.pending.begin(), q.pending.begin() + q.commit);
      assert(q.base == q.draft_history);
      q.pending.clear(); q.commit = 0;
    }
    auto history = q.base;
    for (int i = row; i < end; ++i) {
      assert(positions[i] == (int)history.size());
      history.push_back(tokens[i]); out[i] = next_token(history);
    }
    if (q.keep) q.pending.assign(tokens + row, tokens + end);
    else q.base = history;
    if (mock.cancel_in_step && q.keep && mock.cancel_request) { mock.cancel_request->cancel = true; mock.cancel_in_step = false; }
    row = end;
  }
  return 0;
}
int h5_step_visual(void* e, int rows, const int* slots, const int* tokens, const int* positions, const int* rope, const float*, int* out) {
  mock.visual_steps++;
  for (int r = 0; r < rows; ++r) for (int k = 0; k < 4; ++k) assert(rope[4*r+k] == positions[r]);
  return h5_step(e, rows, slots, tokens, positions, out);
}
int h5_step_head(void* e, int rows, const int* slots, const int* tokens, const int* positions, int* targets,
                 H5HeadMode mode, int count, const int* selected) {
  return h5_step_visual_head(e, rows, slots, tokens, positions, nullptr, nullptr, targets, mode, count, selected);
}
int h5_step_visual_head(void* e, int rows, const int* slots, const int* tokens, const int* positions,
                        const int* rope, const float* visual, int* targets, H5HeadMode mode, int count, const int* selected) {
  h5::HeadDemand demand = h5::HeadDemand::all();
  const auto error = h5::HeadDemand::from_api(mode, count, selected, rows, demand);
  if (error != H5_HEAD_OK) return error;
  std::vector<int> all(rows), compact(demand.count(rows));
  const int status = rope ? h5_step_visual(e, rows, slots, tokens, positions, rope, visual, all.data())
                          : h5_step(e, rows, slots, tokens, positions, all.data());
  if (status) return status;
  for (int j = 0; j < demand.count(rows); ++j) compact[j] = all[demand.source(j)];
  demand.publish(rows, compact.data(), targets);
  mock.no_head_steps += demand.count(rows) == 0;
  mock.partial_head_steps += demand.count(rows) > 0 && demand.count(rows) < rows;
  return 0;
}
int h5_draft(void*, int count, const int* slots, const int* ingest, const int* row0, const int*, const int* anchor, const int* start, int* drafts) {
  int b = 0;
  for (int i = 0; i < count; ++i) {
    auto& q = mock.slots.at(slots[i]); auto history = q.base;
    if (q.keep) history.insert(history.end(), mock.captured.begin() + row0[i], mock.captured.begin() + row0[i] + ingest[i]);
    q.draft_history = history;
    if (anchor[i] < 0) continue;
    assert(start[i] == (int)history.size()); history.push_back(anchor[i]);
    for (int k = 0; k < mock.proposals; ++k) {
      int t = next_token(history); if (k == 1) t = 40;
      drafts[b * mock.proposals + k] = t; history.push_back(t);
    }
    ++b;
  }
  mock.confidence.assign(b * mock.proposals, 2.f); return 0;
}
const float* h5_draft_conf(void*) { return mock.confidence.data(); }
int h5_clone_slot(void*, int src, int dst, int npos) {
  assert(src != mock.donor && dst != mock.donor);
  assert(npos == (int)mock.slots[src].draft_history.size());
  mock.slots[dst] = mock.slots[src]; return 0;
}
int h5_draft_two(void* e, int count, const int* slots, const int* ingest, const int* row0, const int* pos,
                 const int* anchor, const int* start, int* drafts, int* alternatives, int* forks, int) {
  assert(count == 1 && mock.proposals == 7);
  h5_draft(e, count, slots, ingest, row0, pos, anchor, start, drafts);
  auto history = mock.slots[slots[0]].draft_history; history.push_back(anchor[0]);
  for (int k = 0; k < 7; ++k) { int t = next_token(history); alternatives[k] = t; history.push_back(t); }
  assert(alternatives[0] == drafts[0] && alternatives[1] != drafts[1]); forks[0] = 1; return 0;
}
}
static std::shared_ptr<Request> request(std::vector<int> prompt, int B, bool image = false) {
  auto r = std::make_shared<Request>(); r->prompt = std::move(prompt); r->max_new = 12;
  r->prefix_histogram.resize(B+1); r->position_proposed.resize(B); r->position_reached.resize(B);
  r->position_accepted.resize(B); r->position_emitted.resize(B+1);
  r->position_reject_target_eos.resize(B); r->position_reject_proposed_eos.resize(B); r->position_reject_other.resize(B);
  if (image) {
    r->visual.image_tokens = 1; r->visual.next_pos = r->prompt.size();
    r->visual.embedding_row.assign(r->prompt.size(), -1);
    for (int p = 0; p < (int)r->prompt.size(); ++p) r->visual.rope.push_back({p,p,p,p});
  }
  return r;
}
static void await(const std::shared_ptr<Request>& r) {
  std::unique_lock<std::mutex> lock(r->mu);
  assert(r->cv.wait_for(lock, std::chrono::seconds(2), [&] { return r->done && r->settled; }));
}
static void check_greedy(const std::shared_ptr<Request>& r) {
  auto history = r->prompt; std::string text;
  for (int i = 0; i < r->max_new; ++i) { int t = next_token(history); text.push_back('a' + t % 26); history.push_back(t); }
  assert(r->finish == "length" && r->completion_tokens == r->max_new && r->text == text);
}
static void scenario(int B, bool fair, bool prefix, bool adaptive, bool tree = false) {
  alarm(4); mock.proposals = B;
  auto s = new Scheduler; s->e = &mock; s->tk = new Tok("mock"); s->S = adaptive ? 1 : 3;
  s->ctx = 4096; s->budget = 128; s->proposals = B; s->tree2 = tree;
  if (const char* mode = getenv("H5_HEAD_ROWS")) assert(h5::parse_head_mode(mode, s->head_mode) == H5_HEAD_OK);
  assert(h5::serving::configure(fair ? "1" : "0", nullptr, prefix ? "build/serving-prefix.txt" : nullptr, nullptr,
    s->S + (prefix ? 1 : 0), tree, s->serving) == h5::serving::Error::None);
  mock.slots.resize(s->serving.physical_slots); mock.donor = s->serving.template_slot;
  h5::commit::Options options;
  if (adaptive) { options.mode = "confidence"; options.minimum = "0"; options.maximum = std::to_string(B); options.threshold = "1"; }
  assert(h5::commit::configure(options, B, s->commitment) == h5::commit::Error::None);
  if (prefix) {
    std::ofstream file(s->serving.prefix_file); file << "abc"; file.close();
    assert(s->build_prefix());
    if (!adaptive) { assert(s->check_prefix()); assert(s->prefix_checked && s->prefix_target_changes == 0 && s->prefix_draft_changes == 0); }
  }
  const auto donor_before = prefix ? mock.slots[mock.donor].base : std::vector<int>{};
  std::thread([s] { s->run(); }).detach();
  auto hit = request(prefix ? s->prefix : std::vector<int>{2,3}, B);
  for (int i = 0; i < 257; ++i) hit->prompt.push_back(2 + i % 7);
  auto equal = request(prefix ? s->prefix : std::vector<int>{2,3}, B);
  auto miss = request({5,6,7}, B);
  auto queued_cancel = request({1,2}, B); queued_cancel->cancel = true;
  s->submit(hit); s->submit(equal); s->submit(miss); s->submit(queued_cancel);
  for (auto r : {hit,equal,miss}) { await(r); check_greedy(r); }
  await(queued_cancel); assert(queued_cancel->finish == "cancelled");
  if (prefix) assert(s->prefix_hits == 1);
  auto active_cancel = request({2,3}, B); mock.cancel_request = active_cancel; mock.cancel_in_step = true;
  s->submit(active_cancel); await(active_cancel); assert(active_cancel->finish == "cancelled");
  auto reused = request(hit->prompt, B); s->submit(reused); await(reused); check_greedy(reused);
  if (!adaptive) {
    auto image = request(prefix ? hit->prompt : std::vector<int>{2,3}, B, true);
    s->submit(image); await(image); check_greedy(image); assert(mock.visual_steps > 0);
  }
  assert(s->cancelled == 2);
  if (s->head_mode == H5_HEAD_SELECTED) assert(mock.no_head_steps > 0 && mock.partial_head_steps > 0);
  else assert(mock.no_head_steps == 0 && mock.partial_head_steps == 0);
  if (tree) assert(s->tree_rounds > 0 && s->tree_rescues > 0);
  if (prefix) { assert(s->prefix_hits == 2); assert(mock.slots[mock.donor].base == donor_before && mock.slots[mock.donor].draft_history == donor_before); }
  std::_Exit(0);
}
int main() {
  int cases = 0;
  for (int B : {7,16,32,64}) for (bool fair : {false,true}) for (bool prefix : {false,true}) for (bool adaptive : {false,true}) {
    pid_t p = fork(); assert(p >= 0); if (!p) scenario(B,fair,prefix,adaptive);
    int status; assert(waitpid(p,&status,0) == p && WIFEXITED(status) && WEXITSTATUS(status) == 0); ++cases;
  }
  for (bool fair : {false,true}) for (bool prefix : {false,true}) {
    pid_t p = fork(); assert(p >= 0); if (!p) scenario(7,fair,prefix,false,true);
    int status; assert(waitpid(p,&status,0) == p && WIFEXITED(status) && WEXITSTATUS(status) == 0); ++cases;
  }
  printf("actual serving Scheduler contracts pass: %d cases; FIFO/fair128, prefix hit/miss/equal/image, donor isolation, partial rejection/replay, queue/active cancel, reset/reuse, dynamic proposals, adaptive commitment, strict-tree scratch/rescue/clone-back, demand output identity, intermediate/mixed prefill\n", cases);
}
