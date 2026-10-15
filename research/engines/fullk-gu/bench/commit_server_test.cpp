#define main h5_http_main_unused
#include "../src/server.cpp"
#undef main
#include <cassert>
#include <sys/wait.h>
#include <unistd.h>
#include <limits>
#include <fstream>

struct Mock {
  std::vector<int> base, pending, input, visible, prompt;
  std::vector<float> confidence;
  int keep = 0, commit = -1, B = 7, step = 0, eos_position = -1;
  bool invalid_gap = false, inject_mismatch = true, relax = false;
} mock;
static int next_token(const std::vector<int>& history) {
  if (static_cast<int>(history.size()) == mock.eos_position) return EOS1;
  int hash = 0;
  for (int token : history) hash = (hash * 3 + token) % 101;
  return 10 + hash % 20;
}
extern "C" {
void llama_log_set(ggml_log_callback, void*) {}
llama_model_params llama_model_default_params() { return {}; }
llama_model* llama_model_load_from_file(const char*, llama_model_params) { return reinterpret_cast<llama_model*>(1); }
const llama_vocab* llama_model_get_vocab(const llama_model*) { return reinterpret_cast<const llama_vocab*>(1); }
int32_t llama_token_to_piece(const llama_vocab*, llama_token token, char* out, int32_t, int32_t, bool) {
  mock.visible.push_back(token); out[0] = 'x'; return 1;
}
void h5_reset(void*, int) { mock.base.clear(); mock.pending.clear(); mock.input.clear(); mock.visible.clear(); mock.commit = -1; }
void h5_keep(void*, int, int keep) { mock.keep = keep; }
void h5_accept(void*, int, int count) { assert(count > 0 && count <= static_cast<int>(mock.pending.size())); mock.commit = count; }
int h5_step(void*, int rows, const int*, const int* tokens, const int* positions, int* targets) {
  if (mock.keep) {
    if (!mock.pending.empty()) {
      assert(mock.commit >= 1);
      mock.base.insert(mock.base.end(), mock.pending.begin(), mock.pending.begin() + mock.commit);
      mock.pending.clear(); mock.commit = -1;
    }
    auto expected = mock.prompt;
    expected.insert(expected.end(), mock.visible.begin(), mock.visible.end() - 1);
    assert(expected == mock.base);
  }
  mock.input.assign(tokens, tokens + rows);
  auto history = mock.base;
  for (int row = 0; row < rows; ++row) {
    assert(positions[row] == static_cast<int>(history.size()));
    history.push_back(tokens[row]); targets[row] = next_token(history);
    if (mock.relax && mock.keep && row + 1 < rows && !h5::draft_eos(targets[row]) && !h5::draft_eos(tokens[row + 1]))
      targets[row] = tokens[row + 1];
  }
  if (mock.keep) mock.pending = mock.input;
  else mock.base = history;
  mock.step++; return 0;
}
int h5_step_head(void* e, int rows, const int* slots, const int* tokens, const int* positions, int* targets,
                 H5HeadMode mode, int count, const int* selected) {
  h5::HeadDemand demand = h5::HeadDemand::all();
  auto error = h5::HeadDemand::from_api(mode, count, selected, rows, demand);
  if (error != H5_HEAD_OK) return error;
  std::vector<int> all(rows), compact(demand.count(rows));
  const int status = h5_step(e, rows, slots, tokens, positions, all.data());
  if (status) return status;
  for (int j = 0; j < demand.count(rows); ++j) compact[j] = all[demand.source(j)];
  demand.publish(rows, compact.data(), targets);
  return 0;
}
int h5_step_visual_head(void*, int, const int*, const int*, const int*, const int*, const float*, int*, H5HeadMode, int, const int*) { std::abort(); }
int h5_step_visual(void*, int, const int*, const int*, const int*, const int*, const float*, int*) { std::abort(); }
int h5_clone_slot(void*, int, int, int) { std::abort(); }
int h5_copy_prefix(void*, int, int, int) { std::abort(); }
int h5_draft_two(void*, int, const int*, const int*, const int*, const int*, const int*, const int*, int*, int*, int*, int) { std::abort(); }
int h5_draft(void*, int count, const int*, const int* ingest, const int*, const int*, const int* anchor, const int* start, int* drafts) {
  assert(count == 1);
  auto history = mock.base;
  if (mock.keep) history.insert(history.end(), mock.input.begin(), mock.input.begin() + ingest[0]);
  if (anchor[0] < 0) return 0;
  auto expected = mock.prompt;
  expected.insert(expected.end(), mock.visible.begin(), mock.visible.end() - 1);
  assert(history == expected && start[0] == static_cast<int>(history.size()));
  assert(anchor[0] == mock.visible.back());
  history.push_back(anchor[0]);
  mock.confidence.assign(mock.B, 2.f);
  for (int p = 0; p < mock.B; ++p) {
    int token = next_token(history);
    if (mock.inject_mismatch && !h5::draft_eos(token) && mock.step % 2 == 0 && p == 1) token = 40;
    drafts[p] = token; history.push_back(token);
    if (p % 4 == 3) mock.confidence[p] = .25f;
  }
  if (mock.invalid_gap) mock.confidence[0] = std::numeric_limits<float>::quiet_NaN();
  return 0;
}
const float* h5_draft_conf(void*) { return mock.confidence.data(); }
}

static void scenario(int B, const h5::commit::Options& options, bool eos, bool bad_gap, bool relax) {
  alarm(3);
  mock.B = B; mock.prompt = {2, 3}; mock.eos_position = eos ? 17 : -1;
  mock.invalid_gap = bad_gap; mock.inject_mismatch = !eos; mock.relax = relax;
  auto scheduler = new Scheduler;
  scheduler->tk = new Tok("mock"); scheduler->e = &mock;
  scheduler->S = 1; scheduler->ctx = 4096; scheduler->budget = 1; scheduler->proposals = B;
  if (const char* mode = getenv("H5_HEAD_ROWS")) assert(h5::parse_head_mode(mode, scheduler->head_mode) == H5_HEAD_OK);
  assert(h5::commit::configure(options, B, scheduler->commitment) == h5::commit::Error::None);
  auto request = std::make_shared<Request>(); request->prompt = mock.prompt; request->max_new = 40;
  request->prefix_histogram.resize(B + 1); request->position_proposed.resize(B);
  request->position_reached.resize(B); request->position_accepted.resize(B); request->position_emitted.resize(B + 1);
  request->position_reject_target_eos.resize(B); request->position_reject_proposed_eos.resize(B); request->position_reject_other.resize(B);
  std::thread([scheduler] { scheduler->run(); }).detach();
  scheduler->submit(request);
  std::unique_lock<std::mutex> lock(request->mu);
  assert(request->cv.wait_for(lock, std::chrono::seconds(2), [&] { return request->done && request->settled; }));
  if (bad_gap && options.mode != "scheduled") {
    assert(request->finish == "error" && !request->commitment_error.empty());
    assert(request->commitment_rounds.empty() && request->completion_tokens == 1);
  } else {
    assert(request->finish == (eos ? "stop" : "length"));
    assert(request->commitment_error.empty());
    if (options.mode == "fixed") assert(request->commitment_rounds.empty());
    else {
      assert(request->commitment_rounds.size() == static_cast<size_t>(request->verify_rounds));
      long paid = 0, emitted = 1, retained = 0;
      for (const auto& round : request->commitment_rounds) {
        int selected = round["selected_prefix"], accepted = round["accepted_prefix"];
        assert(accepted <= selected && round["verifier_rows"] == selected + 1 && round["committed_inputs"] == accepted + 1);
        assert(round["emitted_tokens"] == round["emitted_drafts"].get<int>() + round["emitted_greedy_tokens"].get<int>());
        assert(round["emitted_greedy_tokens"].get<int>() <= 1);
        assert(round["natural_gate_rejection"].get<bool>() == (accepted < selected && !round["matched_eos"].get<bool>()));
        if (options.mode == "scheduled") {
          const auto& c = scheduler->commitment;
          const int offset = h5::commit::prompt_schedule_offset(c, request->prompt.data(), request->prompt.size());
          const uint64_t walk = offset + round["ordinal"].get<uint64_t>();
          assert(round["schedule_offset"] == offset);
          assert(round["schedule_index"] == walk % h5::commit::SCHEDULE_LENGTH);
          assert(round["schedule_cycle"] == walk / h5::commit::SCHEDULE_LENGTH);
          assert(selected == c.schedule[walk % h5::commit::SCHEDULE_LENGTH]);
          assert(round["selection_trigger"] == "frozen-schedule" && round["selected_gap"].is_null());
        }
        paid += round["verifier_rows"].get<int>(); emitted += round["emitted_tokens"].get<int>(); retained += accepted;
      }
      assert(paid == request->verifier_rows && emitted == request->completion_tokens && retained == request->accepted_drafts);
    }
  }
  if (const char* directory = std::getenv("H5_COMMIT_TEST_OUTPUT")) {
    json response = {{"choices", json::array({{{"message", {{"content", request->text}}}, {"finish_reason", request->finish}}})},
      {"usage", {{"prompt_tokens", 2}, {"completion_tokens", request->completion_tokens}, {"total_tokens", 2 + request->completion_tokens}}},
      {"timings", {{"native_prefill_s", request->native_prefill_s}, {"verify_s", request->native_decode_s}, {"draft_s", request->draft_s},
       {"ttft_s", request->t_first ? request->t_first - request->t_submit : 0.}, {"total_s", request->t_done - request->t_submit}, {"decode_tok_per_s", 0.},
       {"draft_proposals", B}, {"verify_rounds", request->verify_rounds}, {"verifier_rows", request->verifier_rows},
       {"proposed_drafts", request->proposed_drafts}, {"accepted_drafts", request->accepted_drafts},
       {"commitment", {{"config", commitment_config(scheduler->commitment)}, {"error", request->commitment_error.empty() ? json(nullptr) : json(request->commitment_error)},
                       {"rounds", request->commitment_rounds}}}}}};
    const std::string filename = std::string(directory) + "/" + std::to_string(B) + "-" + options.mode + "-" + options.maximum + "-" +
      std::to_string(eos) + "-" + std::to_string(bad_gap) + "-" + std::to_string(relax) + "-" + options.schedule_seed + ".json";
    std::ofstream out(filename);
    assert(out); out << response.dump(2) << '\n'; out.close(); assert(out);
  }
  std::_Exit(0);
}
int main() {
  int cases = 0;
  for (int B : {7, 16, 32, 64}) {
    std::vector<h5::commit::Options> options = {
      {"fixed", "", "", "", ""}, {"periodic", "", "0", "", ""},
      {"periodic", "", std::to_string(B), "", ""},
      {"confidence", "0", std::to_string(B), "1", ""},
      {"feedback", "0", std::to_string(B), "1", "2"}
    };
    if (B == 7) {
      options.push_back({"scheduled", "", "", "", "", "0"});
      options.push_back({"scheduled", "", "", "", "", "20261007"});
    }
    for (const auto& option : options) for (bool eos : {false, true}) for (bool relax : {false, true}) {
      pid_t pid = fork(); assert(pid >= 0);
      if (!pid) scenario(B, option, eos, false, relax);
      int status = 0; assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0); cases++;
    }
    pid_t pid = fork(); assert(pid >= 0);
    if (!pid) scenario(B, {"confidence", "0", std::to_string(B), "1", ""}, false, true, false);
    int status = 0; assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0); cases++;
    if (B == 7) for (bool relax : {false, true}) {
      pid = fork(); assert(pid >= 0);
      if (!pid) scenario(B, {"scheduled", "", "", "", "", "20261007"}, false, true, relax);
      assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0); cases++;
    }
  }
  printf("actual Scheduler mock-engine contracts pass: %d cases; state rollback/ingest/anchor, EOS, cap, telemetry settlement, invalid-gap error\n", cases);
}
