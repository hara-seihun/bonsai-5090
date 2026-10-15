#include "../src/gdn_diagnostic.h"
#include "../tools/gdn_diagnostic_input.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <unistd.h>

using namespace h5;
using namespace h5::diagnostic;

static void require(bool ok, const char* contract) {
  if (!ok) { std::cerr << "FAIL: " << contract << '\n'; std::exit(1); }
}
struct InputFile {
  std::string path;
  InputFile() {
    char name[] = "/tmp/h5-gdn-contract-XXXXXX";
    const int fd = mkstemp(name);
    require(fd >= 0, "create CPU fixture");
    require(close(fd) == 0, "close CPU fixture descriptor");
    path = name;
  }
  ~InputFile() { std::remove(path.c_str()); }
  void set(const std::string& text) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(text.data(), text.size()); file.close();
    require(bool(file), "write CPU fixture");
  }
};

static void decoder_contract() {
  std::vector<float> decoded(GDN_DIAGNOSTIC_HEAD_ELEMENTS);
  std::vector<int8_t> packed(GDN_DIAGNOSTIC_HEAD_ELEMENTS);
  float scales[SS];
  for (int row = 0; row < SS; ++row) {
    scales[row] = (row + 1) * 0.25f;
    for (int col = 0; col < SS; ++col) packed[row * SS + col] = static_cast<int8_t>(-128 + 2 * col);
  }
  require(diagnostic_decode_head(StateFormat::Q8, packed.data(), scales, decoded.data()) == GdnDiagnosticError::None, "Q8 decode succeeds");
  require(decoded[0] == -32 && decoded[64] == 0 && decoded[127] == 31.5f,
          "Q8 signed range includes -128 and positive 126");
  require(decoded[128] == -64 && decoded[255] == 63 && decoded[SS * SS - 1] == 4032,
          "Q8 scales index rows, not columns or heads");
  require(diagnostic_decode_head(StateFormat::Q8, packed.data(), nullptr, decoded.data()) == GdnDiagnosticError::BadArgument,
          "Q8 scales are mandatory");
  require(diagnostic_decode_head(StateFormat::F32, nullptr, nullptr, decoded.data()) == GdnDiagnosticError::BadArgument,
          "payload is mandatory");
  require(diagnostic_decode_head(StateFormat::Q8, packed.data(), scales, nullptr) == GdnDiagnosticError::BadArgument,
          "destination is mandatory");
  require(diagnostic_decode_head(static_cast<StateFormat>(99), packed.data(), scales, decoded.data()) == GdnDiagnosticError::BadArgument,
          "unknown decoder route is rejected");
  std::vector<uint16_t> half(GDN_DIAGNOSTIC_HEAD_ELEMENTS, 0x3c00);
  half[0] = 0x0001; half[1] = 0x03ff; half[2] = 0x8001; half[3] = 0x8000;
  half[4] = 0x0400; half[5] = 0x7bff; half[6] = 0x7c00; half[7] = 0xfc00;
  half[8] = 0x7e00; half[9] = 0x7c01;
  require(diagnostic_decode_head(StateFormat::F16, half.data(), nullptr, decoded.data()) == GdnDiagnosticError::None, "F16 decode succeeds");
  require(decoded[0] == 0x1p-24f && decoded[1] == 1023 * 0x1p-24f && decoded[2] == -0x1p-24f,
          "F16 positive/negative subnormals retain magnitude");
  require(decoded[3] == 0 && std::signbit(decoded[3]), "F16 signed zero retained");
  require(decoded[4] == 0x1p-14f && decoded[5] == 65504 && decoded[10] == 1, "F16 normal exponent limits");
  require(std::isinf(decoded[6]) && decoded[6] > 0 && std::isinf(decoded[7]) && decoded[7] < 0,
          "F16 infinities retained");
  require(std::isnan(decoded[8]) && std::isnan(decoded[9]), "F16 quiet/signaling NaN remain nonfinite");
  half[0] = 0x3f80; half[1] = 0xbf80; half[2] = 0x0001; half[3] = 0x7fc0;
  require(diagnostic_decode_head(StateFormat::BF16, half.data(), nullptr, decoded.data()) == GdnDiagnosticError::None &&
          decoded[0] == 1 && decoded[1] == -1 && decoded[2] == 0x1p-133f && std::isnan(decoded[3]),
          "BF16 route restores high float bits");
  std::vector<float> full(GDN_DIAGNOSTIC_HEAD_ELEMENTS, -2.5f);
  require(diagnostic_decode_head(StateFormat::F32, full.data(), nullptr, decoded.data()) == GdnDiagnosticError::None &&
          decoded.front() == -2.5f && decoded.back() == -2.5f, "F32 route is identity");
}

static void metrics_contract() {
  const float zero[] = {0, 0};
  const float nonzero[] = {3, 4};
  const float perturbed[] = {6, 8};
  auto m = diagnostic_metrics(zero, zero, 2);
  require(m.relative_kind == GdnRelativeKind::ZeroReferenceEqual && m.relative_l2 == 0 && m.absolute_l2 == 0,
          "zero reference/equal candidate has defined zero relative error");
  m = diagnostic_metrics(zero, nonzero, 2);
  require(m.relative_kind == GdnRelativeKind::ZeroReferenceDifferent && std::isinf(m.relative_l2) &&
          m.absolute_l2 == 5 && m.error_max == 4, "zero reference/different candidate has infinite relative error");
  m = diagnostic_metrics(nonzero, perturbed, 2);
  require(m.relative_kind == GdnRelativeKind::Finite && m.reference_l2 == 5 && m.candidate_l2 == 10 &&
          m.absolute_l2 == 5 && m.relative_l2 == 1 && m.error_rms == std::sqrt(12.5),
          "finite head metrics are measured against reference norm");
  const float ref_bad[] = {std::numeric_limits<float>::quiet_NaN(), 1};
  const float cand_bad[] = {0, std::numeric_limits<float>::infinity()};
  m = diagnostic_metrics(ref_bad, cand_bad, 2);
  require(m.relative_kind == GdnRelativeKind::Nonfinite && m.reference_nonfinite == 1 && m.candidate_nonfinite == 1 &&
          std::isnan(m.reference_l2) && std::isnan(m.candidate_l2) && std::isnan(m.reference_rms) &&
          std::isnan(m.candidate_rms) && std::isnan(m.absolute_l2) && std::isnan(m.error_rms) &&
          std::isnan(m.error_max) && std::isnan(m.relative_l2), "nonfinite input invalidates all numeric metrics, retaining counts");
  m = diagnostic_metrics(zero, cand_bad, 2);
  require(m.relative_kind == GdnRelativeKind::Nonfinite && m.reference_nonfinite == 0 && m.candidate_nonfinite == 1 &&
          m.reference_l2 == 0 && m.reference_rms == 0 && std::isnan(m.candidate_l2),
          "candidate-only nonfinite preserves finite reference, not zero-reference divergence");
  m = diagnostic_metrics(nonzero, cand_bad, 2);
  require(m.reference_l2 == 5 && m.reference_rms == std::sqrt(12.5) && std::isnan(m.candidate_l2),
          "finite reference norm uses all reference values even when candidate is nonfinite");
  m = diagnostic_metrics(ref_bad, nonzero, 2);
  require(std::isnan(m.reference_l2) && m.candidate_l2 == 5 && m.candidate_rms == std::sqrt(12.5),
          "finite candidate norm uses all candidate values even when reference is nonfinite");
  m = diagnostic_metrics(nullptr, nullptr, 0);
  require(m.relative_kind == GdnRelativeKind::Nonfinite && std::isnan(m.relative_l2), "empty metrics are explicitly undefined");
}

static void layer_routes_contract() {
  int count = 0;
  for (int layer = 0; layer < NLAYER; ++layer) {
    const int gl = gdn_layer_index(layer);
    if (is_attn(layer)) require(gl == -1, "attention layers have no GDN read route");
    else { require(gl == count && gdn_model_layer(gl) == layer, "GDN layer/model mapping is a bijection"); ++count; }
  }
  require(count == NGDN && gdn_model_layer(-1) == -1 && gdn_model_layer(NGDN) == -1 &&
          gdn_layer_index(-1) == -1 && gdn_layer_index(NLAYER) == -1, "invalid layer routes are rejected");
}

static void environment_routes_contract() {
  struct Arm { const char* name; const char* value; std::optional<std::string> previous; };
  Arm arms[] = {{"H5_STATE_FUSION", "0", {}}, {"H5_MEGA", "0", {}}, {"H5_GDN_SINGLE", "off", {}},
                {"H5_GDN_ARITH", "fp32", {}}, {"H5_GDN_SCALED", "0", {}}, {"H5_GDN_PIPE", "0", {}},
                {"H5_GDN_OCC", "2", {}}, {"H5_TAIL_PREP", "0", {}}, {"H5_GDN_DELTA", "0", {}}};
  for (auto& arm : arms) {
    if (const char* value = std::getenv(arm.name)) arm.previous = value;
    require(unsetenv(arm.name) == 0, "clear diagnostic route environment");
  }
  require(diagnostic_environment() == GdnDiagnosticError::None, "unset environment has ordinary diagnostic route");
  for (const auto& arm : arms) {
    for (const char* invalid : {"", "1", "unknown"}) {
      require(setenv(arm.name, invalid, 1) == 0, "set invalid route option");
      require(diagnostic_environment() == GdnDiagnosticError::UnsupportedRoute,
              "fusion/mega/single/packed/scaled/pipe/occ/tail/delta routes are rejected explicitly");
    }
    require(setenv(arm.name, arm.value, 1) == 0, "set supported route option");
    require(diagnostic_environment() == GdnDiagnosticError::None, "explicit ordinary route options remain supported");
  }
  for (const auto& arm : arms) {
    const int status = arm.previous ? setenv(arm.name, arm.previous->c_str(), 1) : unsetenv(arm.name);
    require(status == 0, "restore route environment");
  }
}

static void parser_contract() {
  InputFile file;
  auto reject = [&](const std::string& text, InputError error, int slots = 2, int context = 16,
                    const std::vector<int>& checkpoints = std::vector<int>{}) {
    file.set(text);
    History out{{Event{true, 123, {8}, {9}, {10}}}, {Checkpoint{777, {{88, 99, 100}}}}, 42};
    const auto result = read_history(file.path, slots, context, checkpoints, out);
    if (result.error != error) {
      std::cerr << "Expected error " << static_cast<int>(error) << ", got " << static_cast<int>(result.error) << ": " << result.message << '\n';
      require(false, "typed parser rejection");
    }
    require(!result.message.empty(), "parser errors name the failed contract");
    require(out.max_rows == 42 && out.events.size() == 1 && out.events[0].reset_slot == 123 &&
            out.events[0].slots == std::vector<int>{8} && out.events[0].tokens == std::vector<int>{9} &&
            out.events[0].positions == std::vector<int>{10} && out.checkpoints.size() == 1 &&
            out.checkpoints[0].event_index == 777 && out.checkpoints[0].slots[0].sequence == 88 &&
            out.checkpoints[0].slots[0].tokens_before == 99 && out.checkpoints[0].slots[0].tokens_after == 100,
            "failure never publishes partial History");
  };
  file.set("RESET 0\nRESET 1\nSTEP 4 1 7 0 1 8 1 0 9 0 0 10 1\nSTEP 2 0 11 2 1 12 2\nRESET 1\nSTEP 1 1 13 0");
  History history{};
  auto result = read_history(file.path, 2, 4, {2, 3, 5, 6}, history);
  require(result.error == InputError::None && result.message.empty(), "grouped multislot ordinary history parses without final newline");
  require(history.events.size() == 6 && history.max_rows == 4 && history.events[2].slots == std::vector<int>({1, 1, 0, 0}),
          "grouped slot order need not be ascending");
  const auto& checkpoints = history.checkpoints;
  require(checkpoints.size() == 4 && checkpoints[0].event_index == 2 && checkpoints[1].event_index == 3,
          "checkpoint indices are sparse 1-based including RESET events");
  require(checkpoints[0].slots[1].sequence == 0 && checkpoints[0].slots[1].tokens_before == 0 && checkpoints[0].slots[1].tokens_after == 0,
          "first reset establishes sequence zero");
  require(checkpoints[1].slots[0].tokens_before == 0 && checkpoints[1].slots[0].tokens_after == 2 &&
          checkpoints[1].slots[1].tokens_before == 0 && checkpoints[1].slots[1].tokens_after == 2,
          "STEP checkpoint reports immediate pre/post counts");
  require(checkpoints[2].slots[0].tokens_before == 3 && checkpoints[2].slots[0].tokens_after == 3 &&
          checkpoints[2].slots[1].sequence == 1 && checkpoints[2].slots[1].tokens_before == 3 && checkpoints[2].slots[1].tokens_after == 0,
          "RESET checkpoint reports previous tokens and new sequence/reset count");
  require(checkpoints[3].slots[1].sequence == 1 && checkpoints[3].slots[1].tokens_before == 0 && checkpoints[3].slots[1].tokens_after == 1,
          "post-reset STEP restarts token positions");
  for (const auto* op : {"KEEP 0 1", "ACCEPT 0 1", "HEADSTEP 1 0 7 0 1 0", "CLONE 0 1", "UNKNOWN"})
    reject(std::string("RESET 0\n") + op + '\n', InputError::UnsupportedEvent);
  for (const auto* malformed : {"", "\n", "RESET\n", "RESET 0 extra\n", "RESET 0 1\n", "RESET 0x\n", "RESET 2147483648\n",
                               "RESET 0\nSTEP\n", "RESET 0\nSTEP 1 0 7\n", "RESET 0\nSTEP 1 0 7 0 extra\n",
                               "RESET 0\nSTEP 1 0 7 0 1\n", "RESET 0\nSTEP 1 0 7x 0\n", "RESET 0\nSTEP 1 0 7 0\n\n"})
    reject(malformed, InputError::MalformedEvent);
  reject(std::string("RESET 0\0junk", 12), InputError::MalformedEvent);
  reject("RESET 0\nSTEP 1 0 -1 0\n", InputError::TokenRange);
  reject("RESET 0\nSTEP 1 0 " + std::to_string(VOCAB) + " 0\n", InputError::TokenRange);
  reject("RESET -1\n", InputError::SlotRange);
  reject("RESET 2\n", InputError::SlotRange);
  reject("RESET 0\nSTEP 1 -1 7 0\n", InputError::SlotRange);
  reject("RESET 0\nSTEP 1 2 7 0\n", InputError::SlotRange);
  reject("RESET 0\nSTEP 1 0 7 -1\n", InputError::PositionRange);
  reject("RESET 0\nSTEP 1 0 7 16\n", InputError::PositionRange);
  reject("STEP 1 0 7 0\n", InputError::MissingReset);
  reject("RESET 0\nSTEP 1 1 7 0\n", InputError::MissingReset);
  reject("RESET 0\n", InputError::MissingReset, 2, 16, {1});
  reject("RESET 0\nRESET 1\nSTEP 3 0 7 0 1 8 0 0 9 1\n", InputError::NoncontiguousSlot);
  reject("RESET 0\nSTEP 1 0 7 1\n", InputError::NonsequentialPosition);
  reject("RESET 0\nSTEP 2 0 7 0 0 8 2\n", InputError::NonsequentialPosition);
  reject("RESET 0\nSTEP 1 0 7 0\nSTEP 1 0 8 0\n", InputError::NonsequentialPosition);
  reject("RESET 0\nSTEP 1 0 7 0\nRESET 0\nSTEP 1 0 8 1\n", InputError::NonsequentialPosition);
  reject("RESET 0\nSTEP 0\n", InputError::LimitExceeded);
  reject("RESET 0\nSTEP 521\n", InputError::LimitExceeded);
  reject(std::string(MAX_EVENT_LINE_BYTES + 1, ' '), InputError::LimitExceeded);
  reject("RESET 0\n", InputError::BadArgument, 0);
  reject("RESET 0\n", InputError::BadArgument, 1, 0);
  reject("RESET 0\n", InputError::BadArgument, 1, 16, {0});
  reject("RESET 0\n", InputError::BadArgument, 1, 16, {1, 1});
  reject("RESET 0\n", InputError::BadArgument, 1, 16, {2, 1});
  reject("RESET 0\n", InputError::MissingCheckpoint, 1, 16, {2});
  reject("RESET 0\n", InputError::LimitExceeded, 1000001);
  reject("RESET 0\n", InputError::LimitExceeded, 1, 16, {1000001});
  reject("RESET 0\n", InputError::LimitExceeded, 1000000, 16, {1, 2, 3, 4, 5});
  result = read_history(file.path + "/missing", 1, 16, {}, history);
  require(result.error == InputError::Io, "file open failure is typed IO");
  result = read_history("/tmp", 1, 16, {}, history);
  require(result.error == InputError::Io, "file read failure is typed IO");
  result = read_history("", 1, 16, {}, history);
  require(result.error == InputError::BadArgument, "unset file path is rejected");
  file.set("\tRESET\t0\r\nSTEP 1 0 " + std::to_string(VOCAB - 1) + " 0\r\n");
  result = read_history(file.path, 1, 1, {1, 2}, history);
  require(result.error == InputError::None && history.events[1].tokens[0] == VOCAB - 1 &&
          history.checkpoints[1].slots[0].tokens_after == 1, "CRLF/whitespace and upper vocabulary/context endpoints are valid");
}

static void history_bounds_contract() {
  InputFile file;
  {
    std::ofstream output(file.path, std::ios::trunc);
    for (size_t i = 0; i <= MAX_HISTORY_EVENTS; ++i) output << "RESET 0\n";
    output.close(); require(bool(output), "write event-capacity fixture");
  }
  History out{{}, {}, 42};
  auto result = read_history(file.path, 1, 520, {}, out);
  require(result.error == InputError::LimitExceeded && out.max_rows == 42 && out.events.empty(), "million-event bound is enforced transactionally");
  std::string step = "STEP 520";
  for (int position = 0; position < 520; ++position) step += " 0 7 " + std::to_string(position);
  step += '\n';
  {
    std::ofstream output(file.path, std::ios::trunc);
    for (size_t rows = 0; rows <= MAX_HISTORY_ROWS; rows += 520) output << "RESET 0\n" << step;
    output.close(); require(bool(output), "write row-capacity fixture");
  }
  result = read_history(file.path, 1, 520, {}, out);
  require(result.error == InputError::LimitExceeded && out.max_rows == 42 && out.events.empty(), "four-million-row bound is enforced transactionally");
  file.set("RESET 0\n" + step);
  result = read_history(file.path, 1, 520, {2}, out);
  require(result.error == InputError::None && out.max_rows == 520 && out.checkpoints[0].slots[0].tokens_after == 520,
          "maximum event rows are valid");
}

static void real_fixture_contract(const std::string& path) {
  History history{};
  auto result = read_history(path, 1, std::numeric_limits<int>::max(), {}, history);
  if (result.error != InputError::None) { std::cerr << result.message << '\n'; require(false, "ordinary real-token fixture parses"); }
  const size_t events = history.events.size();
  require(events >= 3 && history.max_rows > 0, "real-token fixture has resets/prefill/decode");
  result = read_history(path, 1, std::numeric_limits<int>::max(), {1, 2, static_cast<int>(events)}, history);
  require(result.error == InputError::None && history.checkpoints.size() == 3 &&
          history.checkpoints[0].slots[0].sequence == 0 && history.checkpoints[0].slots[0].tokens_after == 0 &&
          history.checkpoints[1].slots[0].tokens_before == 0 && history.checkpoints[1].slots[0].tokens_after == history.max_rows,
          "real-token sparse checkpoint metadata includes first prefill");
  std::cout << "real fixture: " << path << ", events=" << events << ", max_rows=" << history.max_rows
            << ", final_tokens=" << history.checkpoints.back().slots[0].tokens_after << '\n';
}
static void readout_contract() {
  require(diagnostic_shadow_driver_format(StateFormat::F32) && diagnostic_shadow_driver_format(StateFormat::F16) &&
          !diagnostic_shadow_driver_format(StateFormat::Q8) && !diagnostic_shadow_driver_format(StateFormat::BF16), "explicit F32/F16 shadow driver format contract");
  const GdnReadoutSample absent{};
  require(absent.observation == GdnReadoutObservation::Unobserved && absent.row == -1 && absent.token == -1 && absent.position == -1 && std::isnan(absent.max_abs),
          "reset/no executed readout has an explicit unobserved state, not a zero maximum");
  const float finite[] = {65504.f, 70000.f, -1.e30f, 0.f};
  const auto observed = diagnostic_readout_stats(finite, 4, 281, 7, 33048);
  require(observed.observation == GdnReadoutObservation::Observed && observed.row == 281 && observed.token == 7 && observed.position == 33048 &&
          observed.elements == 4 && observed.nonfinite == 0 && observed.max_abs == (double)std::fabs(finite[2]),
          "float readout above F16 range remains finite and preserves exact sample coordinates");
  const float infinite[] = {1.f, -std::numeric_limits<float>::infinity(), 0.f};
  const auto inf = diagnostic_readout_stats(infinite, 3, 0, 0, 0);
  require(inf.nonfinite == 1 && inf.infinity == 1 && inf.nan == 0 && std::isinf(inf.max_abs), "infinite readout maxima stay infinite with nonfinite counts");
  const float nan_values[] = {std::numeric_limits<float>::quiet_NaN(), 2.f, std::numeric_limits<float>::infinity()};
  const auto nan = diagnostic_readout_stats(nan_values, 3, 0, 0, 0);
  require(nan.nonfinite == 2 && nan.nan == 1 && nan.infinity == 1 && std::isnan(nan.max_abs), "NaN readout is not reduced to a finite-only maximum");
  const auto invalid = diagnostic_readout_stats(nullptr, 0, -1, -1, -1);
  require(invalid.observation == GdnReadoutObservation::Unobserved && std::isnan(invalid.max_abs), "absent input is unobserved");
  std::vector<float> raw(GDN_DIAGNOSTIC_HEAD_ELEMENTS, 70000.f), decoded(GDN_DIAGNOSTIC_HEAD_ELEMENTS);
  require(diagnostic_decode_head(StateFormat::F32, raw.data(), nullptr, decoded.data()) == GdnDiagnosticError::None && decoded == raw,
          "F32 recurrent state export keeps values above F16 range without narrowing");
}
int main(int argc, char** argv) {
  require(argc <= 2, "usage: gdn-diagnostic-contract [ordinary-real-event-file]");
  decoder_contract(); metrics_contract(); layer_routes_contract(); environment_routes_contract(); readout_contract(); parser_contract(); history_bounds_contract();
  if (argc == 2) real_fixture_contract(argv[1]);
  std::cout << "GDN diagnostic CPU contracts passed\n";
}
