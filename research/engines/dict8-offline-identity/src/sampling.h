#pragma once
#include "nlohmann/json.hpp"
#include "sampling_contract.h"
#include "sampling_noise.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace h5::sampling {
struct Config {
  double temperature = 0, top_p = 1, min_p = 0, presence_penalty = 0, repetition_penalty = 1;
  int top_k = 0;
  uint64_t seed = 0;
  bool ordinary() const { return temperature > 0 || presence_penalty != 0 || repetition_penalty != 1; }
};
inline bool parse(const nlohmann::ordered_json& body, Config& c, std::string& error) {
  auto number = [&](const char* key, double& out, double low, double high, bool exclusive_low = false) {
    if (!body.contains(key)) return true;
    const auto& value = body[key];
    if (!value.is_number()) { error = std::string(key) + " must be numeric"; return false; }
    out = value.get<double>();
    if (!std::isfinite(out) || out > high || (exclusive_low ? out <= low : out < low)) {
      error = std::string(key) + " outside supported range"; return false;
    }
    return true;
  };
  if (!number("temperature", c.temperature, 0, 2) || !number("top_p", c.top_p, 0, 1, true) ||
      !number("min_p", c.min_p, 0, 1) || !number("presence_penalty", c.presence_penalty, -2, 2) ||
      !number("repetition_penalty", c.repetition_penalty, 0, 2, true)) return false;
  if (body.contains("top_k")) {
    const auto& value = body["top_k"];
    if (!value.is_number_integer() || value.get<double>() < 0 || value.get<double>() > 128) {
      error = "top_k must be an integer in [0,128]"; return false;
    }
    c.top_k = value.get<int>();
  }
  if (body.contains("seed")) {
    const auto& value = body["seed"];
    if (!value.is_number_integer() || value.get<double>() < 0 || value.get<double>() > UINT32_MAX) {
      error = "seed must be an integer in [0,4294967295]"; return false;
    }
    c.seed = value.get<uint64_t>();
  } else c.seed = uint32_t(std::random_device{}());
  return true;
}
inline bool parse_draft_temperature_scale(const char* value, double& scale, std::string& error) {
  if (!value) { scale = 1; return true; } // Unset means inherit the target temperature.
  char* end = nullptr; errno = 0;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0' || errno == ERANGE || !std::isfinite(parsed) || parsed <= 0 || parsed > 1) {
    error = "H5_DRAFT_TEMPERATURE_SCALE must be numeric in (0,1]"; return false;
  }
  scale = parsed; return true;
}
inline bool parse_draft(const nlohmann::ordered_json& body, const Config& target,
                        int proposals, Config& draft, int& verify_limit, std::string& error,
                        double temperature_scale = 1) {
  if (proposals < 1) { error = "configured proposal count must be positive"; return false; }
  if (!std::isfinite(temperature_scale) || temperature_scale <= 0 || temperature_scale > 1) {
    error = "invalid configured draft temperature scale"; return false;
  }
  draft = target; draft.temperature *= temperature_scale; verify_limit = proposals;
  if (body.contains("draft_verify_limit")) {
    const auto& value = body["draft_verify_limit"];
    if (!value.is_number_integer() || value.get<double>() < 0 || value.get<double>() > proposals) {
      error = "draft_verify_limit must be an integer in [0," + std::to_string(proposals) + "]"; return false;
    }
    verify_limit = value.get<int>();
  }
  if (!body.contains("draft_sampling")) return true;
  if (!target.ordinary()) { error = "draft_sampling requires sampled target decoding"; return false; }
  const auto& requested = body["draft_sampling"];
  if (!requested.is_object()) { error = "draft_sampling must be an object"; return false; }
  for (const auto& item : requested.items()) {
    const auto& k = item.key();
    if (k != "temperature" && k != "top_p" && k != "top_k" && k != "min_p" &&
        k != "presence_penalty" && k != "repetition_penalty") {
      error = "unsupported draft_sampling field: " + k; return false;
    }
  }
  auto parameters = requested; parameters["seed"] = target.seed;
  if (!parse(parameters, draft, error)) { error = "draft_sampling: " + error; return false; }
  return true;
}
inline H5SamplingConfig native_config(const Config& c) {
  return {c.temperature, c.top_p, c.min_p, c.presence_penalty, c.repetition_penalty, c.top_k};
}
struct Candidate { int id; double score, probability; };
struct Result { int token; std::string error; };
inline Result select(const float* logits, int vocab, const Config& c,
                     const std::vector<unsigned char>& seen, uint64_t seed, uint64_t index,
                     const std::vector<int>& prefix = {}) {
  if (!logits || vocab < 1 || seen.size() != size_t(vocab)) return {-1, "invalid sampling dimensions"};
  for (const int token : prefix) if (token < 0 || token >= vocab) return {-1, "invalid sampling prefix token"};
  std::vector<Candidate> candidates; candidates.reserve(vocab);
  for (int i = 0; i < vocab; ++i) {
    double score = logits[i];
    if (!std::isfinite(score)) return {-1, "nonfinite target logits"};
    if (seen[i] || std::find(prefix.begin(), prefix.end(), i) != prefix.end()) {
      score = score < 0 ? score * c.repetition_penalty : score / c.repetition_penalty;
      score -= c.presence_penalty;
    }
    candidates.push_back({i, score, 0});
  }
  auto better = [](const Candidate& a, const Candidate& b) { return a.score > b.score || (a.score == b.score && a.id < b.id); };
  if (c.temperature == 0) return {std::min_element(candidates.begin(), candidates.end(), better)->id, {}};
  const int k = c.top_k ? std::min(c.top_k, vocab) : vocab;
  std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), better);
  candidates.resize(k);
  const double best = candidates.front().score;
  double total = 0;
  for (auto& item : candidates) { item.probability = std::exp((item.score - best) / c.temperature); total += item.probability; }
  if (c.min_p > 0) {
    auto end = std::find_if(candidates.begin(), candidates.end(), [&](const Candidate& item) { return item.probability < c.min_p; });
    candidates.erase(end, candidates.end());
    total = 0; for (const auto& item : candidates) total += item.probability;
  }
  double sum = 0; size_t retained = 0;
  do { sum += candidates[retained++].probability; } while (retained < candidates.size() && sum < total * c.top_p);
  int selected = -1; double maximum = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < retained; ++i) {
    const auto& item = candidates[i];
    const double score = (item.score - best) / c.temperature + noise_gumbel(seed, index, item.id);
    if (score > maximum || (score == maximum && item.id < selected)) { maximum = score; selected = item.id; }
  }
  return {selected, {}};
}
}
