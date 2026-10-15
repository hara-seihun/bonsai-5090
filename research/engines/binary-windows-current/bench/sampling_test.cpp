#include "sampling.h"
#include <cassert>
#include <iostream>
#include <numeric>

using namespace h5::sampling;

static std::vector<float> scores(const std::vector<int>& history) {
  const int s = std::accumulate(history.begin(), history.end(), 0);
  return {float(s % 5) * .21f, .8f, -.4f, float(history.size() % 4) * .3f};
}
static int draw(const std::vector<int>& history, const Config& c, uint64_t seed, bool draft) {
  auto logits = scores(history);
  if (draft) { logits[0] += .35f; logits[1] -= .12f; }
  std::vector<unsigned char> seen(4,0);
  for (int t : history) seen[t] = 1;
  const auto selected = select(logits.data(),4,c,seen,seed,history.size());
  assert(selected.error.empty()); return selected.token;
}

int main() {
  Config c; std::string error;
  assert(parse(nlohmann::ordered_json{{"temperature",0},{"seed",42}}, c, error));
  assert(!c.ordinary());
  Config automatic; std::string automatic_error;
  assert(parse(nlohmann::ordered_json::object(), automatic, automatic_error));
  assert(automatic.seed <= UINT32_MAX);
  std::vector<unsigned char> seen(4,0);
  float logits[]{1,2,3,4};
  assert(select(logits,4,c,seen,42,0).token == 3);
  c.presence_penalty=1.5; seen[3]=1;
  assert(select(logits,4,c,seen,42,0).token == 2);
  seen[3]=0;
  assert(select(logits,4,c,seen,42,0,{3}).token == 2);
  c.presence_penalty=0; c.temperature=1; c.top_k=1;
  for (int i=0;i<32;++i) assert(select(logits,4,c,seen,42,i).token == 3);
  c.top_k=0; c.min_p=1;
  assert(select(logits,4,c,seen,42,0).token == 3);
  c.min_p=0; c.top_p=.01;
  assert(select(logits,4,c,seen,42,0).token == 3);
  c.top_p=1;
  for (int i=0;i<64;++i) assert(select(logits,4,c,seen,77,i).token == select(logits,4,c,seen,77,i).token);

  int counts[4]{}; constexpr int trials=40000;
  double total=0;
  for (float x : logits) total += std::exp(double(x)-4);
  for (int seed=0;seed<trials;++seed) {
    const double u=noise_uniform(seed,32,3);
    assert(u>0 && u<1);
    counts[select(logits,4,c,seen,seed,32).token]++;
  }
  for (int i=0;i<4;++i)
    assert(std::abs(double(counts[i])/trials-std::exp(double(logits[i])-4)/total)<.012);

  double scale;
  assert(parse_draft_temperature_scale(nullptr,scale,error) && scale==1);
  assert(parse_draft_temperature_scale("0.5",scale,error) && scale==.5);
  for (const char* value : {"", "0", "-1", "1.1", "nan", "inf", "0.5junk", "1e-999"}) {
    std::string message;
    assert(!parse_draft_temperature_scale(value,scale,message)); assert(!message.empty());
  }
  Config proposal; int verify_limit;
  c.seed=42;
  assert(parse_draft(nlohmann::ordered_json::object(),c,7,proposal,verify_limit,error,.5));
  assert(proposal.temperature==.5 && c.temperature==1 && proposal.seed==c.seed);
  assert(parse_draft(nlohmann::ordered_json{{"draft_sampling",{{"temperature",.8}}}},c,7,proposal,verify_limit,error,.5));
  assert(proposal.temperature==.8);
  assert(parse_draft(nlohmann::ordered_json{{"draft_sampling",{{"top_p",1}}}},c,7,proposal,verify_limit,error,.5));
  assert(proposal.temperature==.5 && proposal.top_p==1);
  Config greedy;
  assert(parse_draft(nlohmann::ordered_json::object(),greedy,7,proposal,verify_limit,error,.5));
  assert(proposal.temperature==0 && !proposal.ordinary());
  for (double invalid_scale : {0.,-1.,2.,std::numeric_limits<double>::quiet_NaN()}) {
    std::string message;
    assert(!parse_draft(nlohmann::ordered_json::object(),c,7,proposal,verify_limit,message,invalid_scale));
    assert(!message.empty());
  }
  assert(parse_draft(nlohmann::ordered_json{{"draft_verify_limit",3},{"draft_sampling",{{"temperature",.5},{"top_p",1},{"min_p",0}}}}, c, 15, proposal, verify_limit, error));
  assert(verify_limit==3 && proposal.temperature==.5 && proposal.top_p==1 && proposal.min_p==0 && proposal.seed==42);
  assert(c.temperature==1 && c.seed==42);
  assert(parse_draft(nlohmann::ordered_json{{"draft_verify_limit",0}},c,15,proposal,verify_limit,error));
  assert(verify_limit==0);
  for (auto body : {nlohmann::ordered_json{{"draft_verify_limit",-1}},nlohmann::ordered_json{{"draft_verify_limit",16}},
      nlohmann::ordered_json{{"draft_verify_limit",true}},nlohmann::ordered_json{{"draft_sampling",0}},
      nlohmann::ordered_json{{"draft_sampling",{{"seed",99}}}},nlohmann::ordered_json{{"draft_sampling",{{"temperature",-1}}}}}) {
    std::string message;
    assert(!parse_draft(body,c,15,proposal,verify_limit,message)); assert(!message.empty());
  }

  long accepted=0, rounds=0;
  c.top_k=4; c.top_p=.95; c.min_p=.05; c.presence_penalty=1.5; c.repetition_penalty=1.1;
  for (uint64_t seed=0;seed<128;++seed) {
    std::vector<int> serial, speculative;
    for (int i=0;i<96;++i) serial.push_back(draw(serial,c,seed,false));
    while (speculative.size()<serial.size()) {
      ++rounds;
      std::vector<int> draft_prefix=speculative, proposals;
      for (int i=0;i<7;++i) {
        const int token=draw(draft_prefix,c,seed,true);
        proposals.push_back(token); draft_prefix.push_back(token);
      }
      for (int i=0;i<=7 && speculative.size()<serial.size();++i) {
        const int target=draw(speculative,c,seed,false);
        speculative.push_back(target);
        if (i==7 || target!=proposals[i]) break;
        ++accepted;
      }
    }
    assert(speculative==serial);
  }
  assert(accepted>rounds);

  logits[0]=std::numeric_limits<float>::quiet_NaN();
  assert(!select(logits,4,c,seen,42,0).error.empty());
  assert(!select(logits,4,c,std::vector<unsigned char>(3),42,0).error.empty());
  for (auto body : {nlohmann::ordered_json{{"temperature",-1}}, nlohmann::ordered_json{{"temperature","hot"}},
       nlohmann::ordered_json{{"top_k",1.5}}, nlohmann::ordered_json{{"top_p",0}},
       nlohmann::ordered_json{{"seed",-1}}, nlohmann::ordered_json{{"seed",4294967296ULL}},
       nlohmann::ordered_json{{"repetition_penalty",0}}}) {
    Config invalid; std::string message;
    assert(!parse(body,invalid,message)); assert(!message.empty());
  }
  std::cout << "sampling contracts passed; coupled accepted=" << accepted << " rounds=" << rounds << '\n';
}
