#include "gdn_diagnostic.h"
#include "gdn_diagnostic_input.h"
#include "gdn_diagnostic_logits.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#ifndef H5_GDN_DIAGNOSTIC_CPU_ONLY
extern "C" {
void* h5_create(const char*, int, int, int);
void h5_reset(void*, int);
void h5_pretune(void*);
int h5_logits(void*, int, float*);
}
#endif

namespace {
using namespace h5;
using namespace h5::diagnostic;
struct Options {
  std::string model, events, output, mode, candidate;
  std::string reference = "f16";
  int slots = 0, context = 0;
  size_t budget_mib = 0;
  bool dry_run = false, reference_explicit = false, readout = false, logits = false;
  std::vector<int> checkpoints, layers, heads;
};
bool fail(const std::string& message) { std::cerr << "gdn-state-drift: " << message << '\n'; return false; }
bool integer(const std::string& text, int& out) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}
bool list(const std::string& text, std::vector<int>& out) {
  std::istringstream input(text); std::string item; int last = -1;
  if (text.empty() || text.back() == ',') return false;
  while (std::getline(input, item, ',')) {
    int value; if (!integer(item, value) || value < 0 || value <= last) return false;
    out.push_back(value); last = value;
  }
  return !out.empty();
}
bool options(int argc, char** argv, Options& o) {
  if (argc < 4) return fail("usage: gdn-state-drift MODEL EVENTS OUTPUT_DIR --mode shadow|pair --candidate q8|f16 --slots N --ctx N --checkpoints EVENT_INDEX,... --layers all|MODEL_LAYER,... --heads all|HEAD,... [--snapshot-budget-mib N (required for pair)] [--reference f32|f16 (omitted: original f16 reference)] [--readout] [--logits (pair only)] [--dry-run]");
  o.model = argv[1]; o.events = argv[2]; o.output = argv[3];
  std::set<std::string> seen;
  for (int i = 4; i < argc; ++i) {
    const std::string name = argv[i];
    if (!seen.insert(name).second) return fail("repeated option " + name);
    if (name == "--dry-run") { o.dry_run = true; continue; }
    if (name == "--readout") { o.readout = true; continue; }
    if (name == "--logits") { o.logits = true; continue; }
    if (i + 1 == argc) return fail("missing value for " + name);
    const std::string value = argv[++i]; int number = 0;
    if (name == "--mode") o.mode = value;
    else if (name == "--candidate") o.candidate = value;
    else if (name == "--reference") { o.reference = value; o.reference_explicit = true; }
    else if (name == "--slots") { if (!integer(value, o.slots) || o.slots < 1 || o.slots > 64) return fail("slots must be 1..64"); }
    else if (name == "--ctx") { if (!integer(value, o.context) || o.context < 1 || o.context > 1048576) return fail("context must be 1..1048576"); }
    else if (name == "--snapshot-budget-mib") { if (!integer(value, number) || number < 1 || number > 65536) return fail("snapshot budget must be 1..65536 MiB"); o.budget_mib = (size_t)number; }
    else if (name == "--checkpoints") { if (!list(value, o.checkpoints) || o.checkpoints.front() < 1 || o.checkpoints.size() > 64) return fail("checkpoints must be 1..64 increasing 1-based event indices"); }
    else if (name == "--layers") {
      if (value == "all") { for (int l = 0; l < NLAYER; ++l) if (!is_attn(l)) o.layers.push_back(l); }
      else if (!list(value, o.layers)) return fail("invalid layer list");
    } else if (name == "--heads") {
      if (value == "all") { for (int h = 0; h < HV; ++h) o.heads.push_back(h); }
      else if (!list(value, o.heads)) return fail("invalid head list");
    } else return fail("unknown option " + name);
  }
  if ((o.mode != "shadow" && o.mode != "pair") || (o.candidate != "q8" && o.candidate != "f16") ||
      !o.slots || !o.context || o.checkpoints.empty() || o.layers.empty() || o.heads.empty()) return fail("mode, candidate, slots, ctx, checkpoints, layers and heads must all be explicit");
  if (o.reference != "f32" && o.reference != "f16") return fail("reference must be f32 or f16");
  if (o.logits && o.mode != "pair") return fail("--logits requires pair mode; shadow alternate logits do not exist");
  for (int l : o.layers) if (gdn_layer_index(l) < 0) return fail("layer is not a physical GDN model layer: " + std::to_string(l));
  for (int h : o.heads) if (h >= HV) return fail("head outside 0..47");
  if ((o.mode == "pair") != (o.budget_mib != 0)) return fail("snapshot budget must be supplied for pair only");
  if (diagnostic_environment() != GdnDiagnosticError::None) return fail("unsupported diagnostic configuration: require separate ordinary FP32 recurrence, GDN_SINGLE=off, STATE_FUSION/MEGA/SCALED/PIPE/TAIL_PREP/DELTA=0, OCC=2");
#ifdef H5_GDN_DIAGNOSTIC_CPU_ONLY
  if (!o.dry_run) return fail("CPU-only input executable requires --dry-run");
#endif
  if (!o.dry_run && (!std::getenv("H5_PLAN_READ") || !*std::getenv("H5_PLAN_READ"))) return fail("runtime requires an explicit frozen H5_PLAN_READ plan shared by both arms");
  return true;
}
std::string quote(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setfill('0') << std::setw(4) << (int)c << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
uint64_t history_hash(const std::string& path) {
  std::ifstream input(path, std::ios::binary); uint64_t hash = 14695981039346656037ULL;
  char buffer[8192];
  while (input) {
    input.read(buffer, sizeof(buffer));
    for (std::streamsize i = 0; i < input.gcount(); ++i) { hash ^= (unsigned char)buffer[i]; hash *= 1099511628211ULL; }
  }
  return hash;
}
bool write_plan(const Options& o, const History& history, size_t bytes) {
  std::ofstream manifest(o.output + "/manifest.json"), samples(o.output + "/samples.csv");
  if (!manifest || !samples) return fail("cannot create manifest/sample plan");
  manifest << "{\n  \"schema\":1,\"dry_run\":" << (o.dry_run ? "true" : "false")
    << ",\"mode\":" << quote(o.mode) << ",\"reference\":" << quote(o.reference) << ",\"reference_explicit\":" << (o.reference_explicit ? "true" : "false")
    << ",\"reference_role\":\"operational reference, not exact ground truth\",\"candidate\":" << quote(o.candidate)
    << ",\"model\":" << quote(o.model) << ",\"events\":" << quote(o.events)
    << ",\"events_fnv1a64\":\"" << std::hex << history_hash(o.events) << std::dec << "\",\"event_count\":" << history.events.size()
    << ",\"slots\":" << o.slots << ",\"requested_context\":" << o.context << ",\"max_rows\":" << history.max_rows
    << ",\"reference_snapshot_bytes\":" << bytes << ",\"snapshot_budget_mib\":" << o.budget_mib
    << ",\"state_shape_per_head\":[128,128],\"head_count\":48,\"gdn_layers\":48"
    << ",\"q8_layout\":\"state_format.h GdnQ8Layer: signed int8 data[HV][SS][SS], float scales[HV][SS], decoded row=data*scale\""
    << ",\"checkpoint_boundary\":\"after whole RESET or STEP event; no split or sampling of generated outputs\""
    << ",\"decay_observed\":false,\"readout_requested\":" << (o.readout ? "true" : "false")
    << ",\"logits_requested\":" << (o.logits ? "true" : "false") << ",\"logits_observer_enabled\":" << (o.logits && !o.dry_run ? "true" : "false")
    << ",\"readout_boundary\":\"last executed row per touched slot of selected STEP, before norm; RESET and untouched slots unobserved\""
    << ",\"logits_boundary\":\"last STEP row only; final compact head chunk index=(rows-1)%128; RESET unobserved\""
    << ",\"logit_metrics\":\"FP64 shifted logsumexp KL(reference||candidate), independent finite top1/top2 and margins; nonfinite distributions rejected; ties lowest vocabulary index\""
#ifdef H5_GDN_DIAGNOSTIC_SOURCE
    << ",\"compiled_source\":" << quote(H5_GDN_DIAGNOSTIC_SOURCE)
    << ",\"engine_compile_flags\":" << quote(H5_GDN_DIAGNOSTIC_FLAGS)
#endif
    << ",\"environment\":{";
  bool first = true;
  for (char** entry = environ; *entry; ++entry) {
    const std::string value = *entry; const auto eq = value.find('=');
    if (value.rfind("H5_", 0) != 0 || eq == std::string::npos) continue;
    if (!first) manifest << ',';
    first = false; manifest << quote(value.substr(0, eq)) << ':' << quote(value.substr(eq + 1));
  }
  manifest << "},\"layers\":[";
  for (size_t i = 0; i < o.layers.size(); ++i) manifest << (i ? "," : "") << o.layers[i];
  manifest << "],\"heads\":[";
  for (size_t i = 0; i < o.heads.size(); ++i) manifest << (i ? "," : "") << o.heads[i];
  manifest << "]\n}\n";
  samples << "event_index,slot,sequence,tokens_before,tokens_after,last_position_after,boundary\n";
  for (const auto& cp : history.checkpoints) for (int slot = 0; slot < o.slots; ++slot) {
    const auto& s = cp.slots[slot];
    samples << cp.event_index << ',' << slot << ',' << s.sequence << ',' << s.tokens_before << ',' << s.tokens_after << ',' << s.tokens_after - 1
      << ',' << (history.events[cp.event_index - 1].reset ? "after_reset" : "after_step") << '\n';
  }
  manifest.close(); samples.close();
  return (manifest && samples) || fail("manifest/sample plan write failed");
}
#ifndef H5_GDN_DIAGNOSTIC_CPU_ONLY
bool check(GdnDiagnosticResult result, const std::string& action) {
  if (result.error == GdnDiagnosticError::None) return true;
  return fail(action + " error=" + std::to_string((int)result.error) + " detail=" + std::to_string(result.detail));
}
const char* relative_name(GdnRelativeKind kind) {
  switch (kind) {
    case GdnRelativeKind::Finite: return "finite";
    case GdnRelativeKind::ZeroReferenceEqual: return "zero_reference_equal";
    case GdnRelativeKind::ZeroReferenceDifferent: return "zero_reference_different";
    case GdnRelativeKind::Nonfinite: return "nonfinite";
  }
  std::abort();
}
void metric_row(std::ostream& out, const Options& o, const Checkpoint& cp, int slot, int layer, int head, const float* ref, const float* candidate) {
  const auto m = diagnostic_metrics(ref, candidate, GDN_DIAGNOSTIC_HEAD_ELEMENTS);
  const auto& s = cp.slots[slot];
  out << o.mode << ',' << o.candidate << ',' << cp.event_index << ',' << slot << ',' << s.sequence << ',' << s.tokens_before << ',' << s.tokens_after
    << ',' << s.tokens_after - 1 << ',' << layer << ',' << gdn_layer_index(layer) << ',' << head << ',' << m.absolute_l2 << ',' << m.relative_l2
    << ',' << relative_name(m.relative_kind) << ',' << m.error_rms << ',' << m.error_max << ',' << m.reference_l2 << ',' << m.candidate_l2
    << ',' << m.reference_rms << ',' << m.candidate_rms << ',' << m.reference_nonfinite << ',' << m.candidate_nonfinite << ',' << o.reference << '\n';
}
std::string number(double value) {
  std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
void logit_row(std::ostream& out, const Options& o, const Checkpoint& cp, const Event& event,
    const float* reference, const float* candidate, int reference_prediction, int candidate_prediction) {
  std::vector<std::string> fields(32);
  fields[0] = o.mode; fields[1] = o.reference; fields[2] = o.candidate; fields[3] = std::to_string(cp.event_index);
  fields[4] = event.reset ? "after_reset" : "after_step"; fields[5] = event.reset ? "unobserved" : "observed";
  if (!event.reset) {
    const size_t row = event.slots.size() - 1;
    const auto m = logit_metrics(reference, candidate, VOCAB);
    fields[6] = std::to_string(row); fields[7] = std::to_string(row % 128);
    fields[8] = std::to_string(event.slots[row]); fields[9] = std::to_string(cp.slots[event.slots[row]].sequence);
    fields[10] = std::to_string(event.tokens[row]); fields[11] = std::to_string(event.positions[row]);
    fields[12] = std::to_string(reference_prediction); fields[13] = std::to_string(candidate_prediction);
    fields[14] = logit_metrics_name(m.kind); fields[15] = logit_metrics_name(m.reference.kind); fields[16] = logit_metrics_name(m.candidate.kind);
    fields[17] = std::to_string(VOCAB); fields[18] = std::to_string(m.reference.finite); fields[19] = std::to_string(m.candidate.finite);
    fields[20] = std::to_string(m.reference.nan); fields[21] = std::to_string(m.candidate.nan);
    fields[22] = std::to_string(m.reference.infinity); fields[23] = std::to_string(m.candidate.infinity);
    if (m.reference.kind == LogitMetricsKind::Finite) {
      fields[24] = std::to_string(m.reference.top1); fields[25] = std::to_string(m.reference.top2); fields[28] = number(m.reference.margin);
    }
    if (m.candidate.kind == LogitMetricsKind::Finite) {
      fields[26] = std::to_string(m.candidate.top1); fields[27] = std::to_string(m.candidate.top2); fields[29] = number(m.candidate.margin);
    }
    if (m.kind == LogitMetricsKind::Finite) {
      fields[30] = m.reference.top1 == m.candidate.top1 ? "1" : "0"; fields[31] = number(m.kl_reference_candidate);
    }
  }
  for (size_t i = 0; i < fields.size(); ++i) out << (i ? "," : "") << fields[i];
  out << '\n';
}
bool readout_rows(std::ostream& out, const Options& o, const Checkpoint& cp, const Event& event,
    void* engine, GdnDiagnosticSource source, const char* arm, const std::string& format) {
  for (int slot = 0; slot < o.slots; ++slot) {
    const auto found = std::find(event.slots.rbegin(), event.slots.rend(), slot);
    const bool touched = !event.reset && found != event.slots.rend();
    for (int layer : o.layers) for (int head : o.heads) {
      GdnReadoutSample sample{};
      if (touched && !check(h5_gdn_readout_read(engine, source, slot, layer, head, &sample), "readout read")) return false;
      if (touched && sample.observation != GdnReadoutObservation::Observed) return fail("selected STEP has no captured readout for a touched slot");
      if (sample.observation == GdnReadoutObservation::Observed) {
        const int row = static_cast<int>(event.slots.size() - 1 - (found - event.slots.rbegin()));
        if (sample.row != row || sample.token != event.tokens[row] || sample.position != event.positions[row]) return fail("readout sample does not identify the selected STEP's last slot row");
      }
      out << o.mode << ',' << o.reference << ',' << o.candidate << ',' << arm << ',' << format << ',' << cp.event_index << ',' << slot
        << ',' << cp.slots[slot].sequence << ',' << layer << ',' << gdn_layer_index(layer) << ',' << head << ',';
      if (sample.observation == GdnReadoutObservation::Unobserved) {
        out << "unobserved"; for (int i = 0; i < 8; ++i) out << ',';
      } else {
        out << "observed," << sample.row << ',' << sample.token << ',' << sample.position << ',' << sample.elements << ',' << sample.nonfinite
          << ',' << sample.nan << ',' << sample.infinity << ',' << sample.max_abs;
      }
      out << '\n';
      if (!out) return fail("readout write failed");
    }
  }
  return true;
}
bool finalize_readouts(const Options& o) {
  std::ofstream output(o.output + "/readouts.csv.part");
  if (!output) return fail("cannot create combined readouts");
  output << "mode,reference,candidate,arm,format,event_index,slot,sequence,model_layer,gdn_layer,head,observation,row,token,position,elements,nonfinite,nan,infinity,max_abs\n";
  const std::vector<std::string> arms = o.mode == "pair" ? std::vector<std::string>{"reference", "candidate"} : std::vector<std::string>{"candidate"};
  for (const auto& arm : arms) {
    const std::string path = o.output + "/readouts-" + arm + ".csv.part";
    std::ifstream input(path); if (!input) return fail("arm readouts unavailable");
    std::string line;
    while (std::getline(input, line)) output << line << '\n';
    if (input.bad() || !output) return fail("combined readout IO failed");
    input.close();
  }
  output.close(); if (!output) return fail("combined readout close failed");
  std::error_code error; std::filesystem::rename(o.output + "/readouts.csv.part", o.output + "/readouts.csv", error);
  if (error) return fail("readout finalization failed");
  for (const auto& arm : arms)
    if (!std::filesystem::remove(o.output + "/readouts-" + arm + ".csv.part", error) || error) return fail("arm readout removal failed");
  return true;
}
std::string reference_path(const Options& o, const Checkpoint& cp) { return o.output + "/reference/" + std::to_string(cp.event_index) + ".f32"; }
int run(const Options& o, const History& history, bool reference_arm) {
  const bool shadow = o.mode == "shadow";
  const std::string state = reference_arm || shadow ? o.reference : o.candidate;
  if (setenv("H5_STATE", state.c_str(), 1)) return fail("set H5_STATE failed"), 1;
  void* engine = h5_create(o.model.c_str(), o.slots, o.context, std::max(1, history.max_rows));
  if (!engine) return fail("model load failed"), 1;
  const auto mode = !shadow ? GdnDiagnosticMode::Observe : o.candidate == "q8" ? GdnDiagnosticMode::ShadowQ8 : GdnDiagnosticMode::ShadowF16;
  if (!check(h5_gdn_diagnostic_configure(engine, mode), "configure")) return 1;
  if (o.readout && !check(h5_gdn_readout_configure(engine, o.layers.size(), o.layers.data(), o.heads.size(), o.heads.data()), "configure readout")) return 1;
  h5_pretune(engine);
  GdnDiagnosticInfo info{};
  if (!check(h5_gdn_diagnostic_info(engine, GdnDiagnosticSource::Driver, &info), "driver info")) return 1;
  std::ofstream actual(o.output + (reference_arm ? "/reference-config.csv" : "/candidate-config.csv"));
  actual << "source,format,actual_context,max_rows,slots\n" << "driver," << state_format_name(info.format) << ',' << info.context << ',' << info.max_rows << ',' << info.slots << '\n';
  if (shadow) {
    if (!check(h5_gdn_diagnostic_info(engine, GdnDiagnosticSource::Shadow, &info), "shadow info")) return 1;
    actual << "shadow," << state_format_name(info.format) << ',' << info.context << ',' << info.max_rows << ',' << info.slots << '\n';
  }
  actual.close(); if (!actual) return fail("actual configuration write failed"), 1;
  std::ofstream metrics;
  if (!reference_arm) {
    metrics.open(o.output + "/metrics.csv.part"); metrics << std::setprecision(17);
    metrics << "mode,candidate,event_index,slot,sequence,tokens_before,tokens_after,last_position_after,model_layer,gdn_layer,head,absolute_l2,relative_l2,relative_kind,error_rms,error_max,reference_l2,candidate_l2,reference_rms,candidate_rms,reference_nonfinite,candidate_nonfinite,reference\n";
    if (!metrics) return fail("cannot create metrics output"), 1;
  }
  std::ofstream readouts, logits;
  if (o.readout) {
    readouts.open(o.output + (reference_arm ? "/readouts-reference.csv.part" : "/readouts-candidate.csv.part")); readouts << std::setprecision(17);
    if (!readouts) return fail("cannot create arm readouts"), 1;
  }
  if (o.logits && !reference_arm) {
    logits.open(o.output + "/logits.csv.part");
    logits << "mode,reference,candidate,event_index,boundary,observation,row,compact_row,slot,sequence,token,position,reference_prediction,candidate_prediction,metrics_kind,reference_kind,candidate_kind,elements,reference_finite,candidate_finite,reference_nan,candidate_nan,reference_infinity,candidate_infinity,reference_top1,reference_top2,candidate_top1,candidate_top2,reference_margin,candidate_margin,top1_agreement,kl_reference_candidate\n";
    if (!logits) return fail("cannot create logit metrics"), 1;
  }
  std::vector<float> ref(GDN_DIAGNOSTIC_HEAD_ELEMENTS), candidate(GDN_DIAGNOSTIC_HEAD_ELEMENTS);
  std::vector<float> reference_logits, candidate_logits;
  if (o.logits) { candidate_logits.resize(VOCAB); if (!reference_arm) reference_logits.resize(VOCAB); }
  std::vector<int> predicted(history.max_rows); size_t checkpoint = 0;
  const size_t last = history.checkpoints.back().event_index;
  for (size_t index = 0; index < last; ++index) {
    const auto& event = history.events[index];
    const bool selected = checkpoint < history.checkpoints.size() && history.checkpoints[checkpoint].event_index == index + 1;
    if (event.reset) h5_reset(engine, event.reset_slot);
    else {
      if (o.readout && selected && !check(h5_gdn_readout_arm(engine), "arm readout")) return 1;
      if (!check(h5_gdn_diagnostic_step(engine, event.slots.size(), event.slots.data(), event.tokens.data(), event.positions.data(), predicted.data()), "STEP event " + std::to_string(index + 1))) return 1;
    }
    if (!selected) continue;
    const auto& cp = history.checkpoints[checkpoint++];
    std::ifstream reference_input; std::ofstream reference_output;
    if (!shadow) {
      if (reference_arm) reference_output.open(reference_path(o, cp), std::ios::binary);
      else reference_input.open(reference_path(o, cp), std::ios::binary);
      if (reference_arm ? !reference_output : !reference_input) return fail("reference checkpoint file unavailable"), 1;
    }
    for (int slot = 0; slot < o.slots; ++slot) for (int layer : o.layers) for (int head : o.heads) {
      if (!check(h5_gdn_diagnostic_read(engine, GdnDiagnosticSource::Driver, slot, layer, head, candidate.data(), candidate.size()), "state read")) return 1;
      if (reference_arm) {
        reference_output.write((const char*)candidate.data(), candidate.size() * sizeof(float));
        if (!reference_output) return fail("reference snapshot write failed"), 1;
        continue;
      }
      if (shadow) {
        ref.swap(candidate);
        if (!check(h5_gdn_diagnostic_read(engine, GdnDiagnosticSource::Shadow, slot, layer, head, candidate.data(), candidate.size()), "shadow read")) return 1;
      } else {
        reference_input.read((char*)ref.data(), ref.size() * sizeof(float));
        if (!reference_input) return fail("reference snapshot truncated"), 1;
      }
      metric_row(metrics, o, cp, slot, layer, head, ref.data(), candidate.data());
      if (!metrics) return fail("metrics write failed"), 1;
    }
    if (o.readout) {
      const char* arm = shadow ? "driver" : reference_arm ? "reference" : "candidate";
      if (!readout_rows(readouts, o, cp, event, engine, GdnDiagnosticSource::Driver, arm, state)) return 1;
      if (shadow && !readout_rows(readouts, o, cp, event, engine, GdnDiagnosticSource::Shadow, "shadow", o.candidate)) return 1;
    }
    if (o.logits) {
      int reference_prediction = -1, candidate_prediction = -1;
      if (!event.reset) {
        const int row = static_cast<int>(event.slots.size()) - 1;
        const int status = h5_logits(engine, row % 128, candidate_logits.data());
        if (status) return fail("checkpoint logits unavailable: status=" + std::to_string(status)), 1;
        candidate_prediction = predicted[row];
        if (reference_arm) {
          reference_output.write((const char*)candidate_logits.data(), candidate_logits.size() * sizeof(float));
          reference_output.write((const char*)&candidate_prediction, sizeof(candidate_prediction));
          if (!reference_output) return fail("reference logit snapshot write failed"), 1;
        } else {
          reference_input.read((char*)reference_logits.data(), reference_logits.size() * sizeof(float));
          reference_input.read((char*)&reference_prediction, sizeof(reference_prediction));
          if (!reference_input) return fail("reference logit snapshot truncated"), 1;
        }
      }
      if (!reference_arm) {
        logit_row(logits, o, cp, event, reference_logits.data(), candidate_logits.data(), reference_prediction, candidate_prediction);
        if (!logits) return fail("logit metrics write failed"), 1;
      }
    }
    if (reference_arm) { reference_output.close(); if (!reference_output) return fail("reference snapshot close failed"), 1; }
    else if (!shadow) {
      if (reference_input.peek() != std::char_traits<char>::eof()) return fail("reference snapshot has trailing bytes"), 1;
      reference_input.close();
      std::error_code error; if (!std::filesystem::remove(reference_path(o, cp), error) || error) return fail("reference snapshot removal failed"), 1;
    }
  }
  if (o.readout) { readouts.close(); if (!readouts) return fail("arm readout close failed"), 1; }
  if (o.logits && !reference_arm) {
    logits.close(); if (!logits) return fail("logit metrics close failed"), 1;
    std::error_code error; std::filesystem::rename(o.output + "/logits.csv.part", o.output + "/logits.csv", error);
    if (error) return fail("logit metrics finalization failed"), 1;
  }
  if (!reference_arm) {
    metrics.close(); if (!metrics) return fail("metrics close failed"), 1;
    std::error_code error; std::filesystem::rename(o.output + "/metrics.csv.part", o.output + "/metrics.csv", error);
    if (error) return fail("metrics finalization failed"), 1;
  }
  return 0;
}
bool child(const Options& o, const History& history, bool reference_arm) {
  const pid_t pid = fork();
  if (pid < 0) return fail("fork failed");
  if (pid == 0) {
    int status = 1;
    try { status = run(o, history, reference_arm); }
    catch (const std::exception& error) { fail(std::string("runtime failure: ") + error.what()); }
    std::cout.flush(); std::cerr.flush();
    _exit(status); // Process exit owns every engine/CUDA allocation; no parent CUDA context exists.
  }
  int status = 0; pid_t result;
  do { result = waitpid(pid, &status, 0); } while (result < 0 && errno == EINTR);
  if (result < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) return fail(reference_arm ? "reference child failed" : "candidate child failed");
  return true;
}
#endif
}
int main(int argc, char** argv) {
  try {
    Options o; if (!options(argc, argv, o)) return 1;
    History history;
    const auto result = read_history(o.events, o.slots, o.context, o.checkpoints, history);
    if (result.error != InputError::None) return fail("input error=" + std::to_string((int)result.error) + " " + result.message), 1;
    size_t bytes = o.mode == "pair" ? history.checkpoints.size() * (size_t)o.slots * o.layers.size() * o.heads.size() * GDN_DIAGNOSTIC_HEAD_ELEMENTS * sizeof(float) : 0;
    if (o.logits) for (const auto& cp : history.checkpoints) if (!history.events[cp.event_index - 1].reset) bytes += (size_t)VOCAB * sizeof(float) + sizeof(int);
    if (bytes > o.budget_mib * 1048576ULL) return fail("sparse reference snapshots require " + std::to_string(bytes) + " bytes, over explicit budget"), 1;
    std::error_code error;
    if (!std::filesystem::create_directory(o.output, error) || error) return fail("output directory must not exist and its parent must exist"), 1;
    if (!write_plan(o, history, bytes)) return 1;
    if (o.dry_run) {
      std::cout << "validated " << history.events.size() << " events, " << history.checkpoints.size() << " checkpoints, " << bytes << " reference bytes; no CUDA calls\n";
      return 0;
    }
#ifndef H5_GDN_DIAGNOSTIC_CPU_ONLY
    if (o.mode == "pair") {
      if (!std::filesystem::create_directory(o.output + "/reference", error) || error) return fail("cannot create reference storage"), 1;
      if (!child(o, history, true)) return 1;
    }
    if (!child(o, history, false)) return 1;
    if (o.readout && !finalize_readouts(o)) return 1;
    if (o.mode == "pair" && (!std::filesystem::remove(o.output + "/reference", error) || error)) return fail("cannot remove empty reference storage"), 1;
    std::ofstream complete(o.output + "/complete"); complete << "complete\n"; complete.close();
    if (!complete) return fail("completion receipt write failed"), 1;
    std::cout << "complete: " << o.output << "/metrics.csv\n";
    return 0;
#else
    return 1;
#endif
  } catch (const std::exception& error) { return fail(std::string("host failure: ") + error.what()), 1; }
}
