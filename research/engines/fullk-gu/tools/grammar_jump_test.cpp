#define main unused_server_main
#include "../src/server.cpp"
#undef main
#include <cassert>
#include <cstdlib>
struct Mock {
  std::vector<int> captured_slot, captured_pos, captured_tok, committed, pending, ring;
  std::vector<uint32_t> masks;
  int keep = 0, accept = 0, calls = 0, heads = 0, drafts = 0;
  std::string expected;
  std::atomic<int> ring_rows{0};
  std::mutex mu;
  std::condition_variable cv;
} mock;
extern "C" {
llama_model_params llama_model_default_params() { return {}; }
llama_model* llama_model_load_from_file(const char*, llama_model_params) { return nullptr; }
void llama_log_set(ggml_log_callback, void*) {}
const llama_vocab* llama_model_get_vocab(const llama_model*) { return nullptr; }
int h5_copy_prefix(void*,int,int,int){abort();}
int32_t llama_token_to_piece(const llama_vocab*, llama_token t, char* buf, int32_t size, int32_t, bool) { assert(size>0); buf[0]=char(t); return 1; }
int h5_grammar_draft_timings(void*, double* body, double* select, double* wait, int* waits) {
  *body = *select = *wait = 0; *waits = 0; return 0;
}
int h5_set_grammar_head_options(void*, int count, const int* thresholds, const int* bulk) {
  if (count < 0 || (count && (!thresholds || !bulk))) return -1;
  for (int i = 0; i < count; ++i) if (thresholds[i] || bulk[i]) return -3;
  return 0;
}
int h5_set_grammar_masks(void*, int rows, const uint32_t* masks) {
  mock.masks.assign(masks, masks + size_t(rows)*h5::GRAMMAR_WORDS); return 0;
}
int h5_step_head(void*, int n, const int* slot, const int* tokens, const int* pos, int* out, H5HeadMode mode, int nh, const int* hr) {
  assert(mode == H5_HEAD_SELECTED);
  if (!mock.pending.empty()) { mock.committed.insert(mock.committed.end(), mock.pending.begin(), mock.pending.begin()+mock.accept); mock.pending.clear(); mock.accept=0; }
  assert(pos[0] == int(mock.committed.size()));
  assert(mock.ring.size() <= mock.committed.size());
  assert(std::equal(mock.ring.begin(), mock.ring.end(), mock.committed.begin()));
  mock.captured_slot.assign(slot, slot+n); mock.captured_pos.assign(pos,pos+n); mock.captured_tok.assign(tokens,tokens+n);
  for (int r=0;r<n;++r) { assert(slot[r]==0 && pos[r]==pos[0]+r); out[r]=-1; }
  auto& next = mock.keep ? mock.pending : mock.committed;
  next.insert(next.end(),tokens,tokens+n);
  for (int h=0;h<nh;++h) {
    int row=hr[h]; const auto* mask=mock.masks.data()+size_t(row)*h5::GRAMMAR_WORDS;
    int index=pos[row]; int desired=index<int(mock.expected.size())?int(mock.expected[index]):EOS1;
    if (!(mask[desired/32] & (1u<<(desired%32)))) {
      desired=-1; for (int w=0;w<h5::GRAMMAR_WORDS;++w) if(mask[w]) { desired=w*32+__builtin_ctz(mask[w]);break; }
    }
    assert(desired>=0); out[row]=desired; ++mock.heads;
  }
  ++mock.calls; return 0;
}
int h5_step_visual_head(void*,int,const int*,const int*,const int*,const int*,const float*,int*,H5HeadMode,int,const int*) { abort(); }
void h5_reset(void*, int) { mock.committed.clear(); mock.pending.clear(); mock.ring.clear(); mock.accept=mock.keep=0; }
void h5_keep(void*, int, int keep) { mock.keep=keep; }
void h5_accept(void*,int,int n) { assert(n<=int(mock.pending.size()));mock.accept=n; }
int h5_draft_grammar(void*, int ns, const int* slot, const int* ni, const int* row0, const int* pos0, const int* anchors, const int*, int* drafts, H5GrammarMask callback, void* ctx) {
  int block=0;
  for(int i=0;i<ns;++i) {
    assert(slot[i]==0 && int(mock.ring.size())==pos0[i]);
    for(int j=0;j<ni[i];++j) {int r=row0[i]+j;assert(mock.captured_slot[r]==slot[i] && mock.captured_pos[r]==pos0[i]+j);mock.ring.push_back(mock.captured_tok[r]);}
    if(anchors[i]>=0) {
      ++mock.drafts; int prev=-1;std::vector<uint32_t>mask(h5::GRAMMAR_WORDS);
      for(int p=0;p<7;++p) { assert(callback(ctx,block,p,prev,mask.data(),h5::GRAMMAR_WORDS)>0);int token=-1;for(int w=0;w<h5::GRAMMAR_WORDS;++w)if(mask[w]){token=w*32+__builtin_ctz(mask[w]);break;}assert(token>=0);drafts[block*7+p]=prev=token; }
      ++block;
    }
  }
  { std::lock_guard<std::mutex> lock(mock.mu); mock.ring_rows = mock.ring.size(); }
  mock.cv.notify_all();
  return 0;
}
int h5_draft(void* e,int n,const int* s,const int* ni,const int* r,const int* p,const int* a,const int* st,int* d) {
  for(int i=0;i<n;++i)assert(a[i]<0);
  return h5_draft_grammar(e,n,s,ni,r,p,a,st,d,nullptr,nullptr);
}
int h5_draft_two(void*,int,const int*,const int*,const int*,const int*,const int*,const int*,int*,int*,int*,int){abort();}
int h5_clone_slot(void*,int,int,int){abort();}
int h5_row_stats(void*,int,h5::RowStat*){abort();}
int h5_set_logit_adj(void*,int,const int*,const float*){abort();}
int h5_set_draft_lz(void*,int,const int*,const int*,const float*){abort();}
const float* h5_draft_conf(void*){static float f[7]={1,1,1,1,1,1,1};return f;}
int h5_sample_target(void*,const H5SamplingConfig*,uint64_t,uint64_t,const unsigned char*,const int*,int,int,int*){abort();}
int h5_draft_sampled(void*,int,const int*,const int*,const int*,const int*,const int*,const int*,int*,const H5SamplingConfig*,uint64_t,uint64_t,const unsigned char*){abort();}
}
int main(int argc,char** argv) {
  setenv("H5_VERIFY_ALL_BELOW", "1", 1);
  setenv("H5_TRACE", "0", 1);
  assert(argc == 3);
  const std::string route = argv[1], scenario = argv[2];
  assert(route == "plain" || route == "spec");
  assert(scenario == "dynamic" || scenario == "const" || scenario == "cap" || scenario == "context");
  const bool spec = route == "spec", dynamic = scenario == "dynamic";
  g_spec = spec;
  std::vector<std::string>pieces(h5::VOCAB);for(int c=32;c<127;++c)pieces[c]=std::string(1,char(c));
  auto vocab=std::make_shared<h5::grammar::Vocabulary>(std::move(pieces),std::vector<int>{EOS1,EOS2});
  h5::grammar::CompileOptions options;options.mode=h5::grammar::Mode::CanonicalFixed;options.encode=[](const std::string&s){return std::vector<int>(s.begin(),s.end());};
  json schema={{"type","object"},{"properties",{{"s",{{"type","string"}}},{"tail",{{"const","uvwxyz0123456789"}}}}},{"required",json::array({"s","tail"})},{"additionalProperties",false}};
  mock.expected="{\"s\":\"abc\",\"tail\":\"uvwxyz0123456789\"}";
  if (!dynamic) { schema = {{"const", "abcdefghijklmnopqrstuvwxyz0123456789"}}; mock.expected = schema["const"].dump(); }
  auto compiled=h5::grammar::compile(schema,vocab,options);assert(compiled);
  auto r=std::make_shared<Request>();r->prompt={999};r->grammar=std::move(compiled.matcher);r->grammar_jump_forward=true;r->grammar_mode=options.mode;r->max_new=100;r->draft_verify_limit=spec?7:0;
  r->prefix_histogram.resize(8);r->position_proposed.resize(7);r->position_reached.resize(7);r->position_accepted.resize(7);r->position_emitted.resize(8);r->position_reject_target_eos.resize(7);r->position_reject_proposed_eos.resize(7);r->position_reject_other.resize(7);
  Scheduler scheduler;scheduler.S=1;scheduler.ctx=1024;scheduler.budget=1; scheduler.grammar_vocab=vocab;Tok tk("mock");scheduler.tk=&tk;
  if (scenario == "cap") r->max_new = 10;
  if (scenario == "context") scheduler.ctx = 20;
  scheduler.submit(r);std::thread([&]{scheduler.run();}).detach();
  std::unique_lock<std::mutex>lock(r->mu);assert(r->cv.wait_for(lock,std::chrono::seconds(5),[&]{return r->done&&r->settled;}));
  const bool capped = scenario == "cap" || scenario == "context";
  const int output = scenario == "cap" ? 10 : scenario == "context" ? 19 : mock.expected.size();
  assert(r->text == mock.expected.substr(0, output) && r->completion_tokens == output);
  assert(r->grammar->complete() == !capped && r->commitment_error.empty());
  assert(r->finish == (capped ? "length" : "stop"));
  assert(r->stop_reason == (scenario == "cap" ? "max_tokens" : scenario == "context" ? "context_limit" : "grammar_complete"));
  assert(r->jumped_tokens > 0 && r->jump_spans >= 2 && r->grammar_forced >= r->jumped_tokens);
  assert(r->jumped_tokens <= r->completion_tokens);
  assert(r->jump_input_rows >= r->jumped_tokens);
  std::vector<int> committed = mock.committed;
  committed.insert(committed.end(), mock.pending.begin(), mock.pending.begin() + mock.accept);
  const int inputs = r->prompt.size() + r->completion_tokens;
  // A branching final draw is emitted but need not be ingested after completion.
  assert(int(committed.size()) == inputs || int(committed.size()) == inputs - 1);
  std::vector<int> expected_inputs = r->prompt;
  expected_inputs.insert(expected_inputs.end(), r->text.begin(), r->text.end());
  assert(std::equal(committed.begin(), committed.end(), expected_inputs.begin()));
  if (spec && !capped) {
    std::unique_lock<std::mutex> ring_lock(mock.mu);
    assert(mock.cv.wait_for(ring_lock, std::chrono::seconds(5), [&] {return mock.ring_rows == int(committed.size());}));
  }
  if (spec && dynamic) assert(r->verify_rounds > 0 && mock.drafts > 0 && r->accepted_drafts < r->proposed_drafts);
  // The v3 law retains ordinary whitespace branches around const values.
  // Only singleton spans bypass heads; branching positions still run them.
  assert(mock.heads > 0);
  if (!spec) assert(mock.drafts == 0 && r->verify_rounds == 0);
  printf("scheduler %s/%s: correct output/caps, %ld jumped in %ld chunks, %ld verify rounds, %d heads, %d target calls; replay/capture/ring contiguous\n", route.c_str(), scenario.c_str(), r->jumped_tokens, r->jump_spans, r->verify_rounds, mock.heads, mock.calls);
  fflush(stdout); std::_Exit(0);
}
