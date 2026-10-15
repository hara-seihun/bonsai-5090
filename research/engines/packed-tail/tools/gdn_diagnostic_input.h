#pragma once
#include "../src/model.h"
#include <algorithm>
#include <charconv>
#include <fstream>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace h5::diagnostic {
enum class InputError {
  None, BadArgument, Io, MalformedEvent, UnsupportedEvent, LimitExceeded,
  SlotRange, TokenRange, PositionRange, MissingReset, NoncontiguousSlot,
  NonsequentialPosition, MissingCheckpoint, HostAllocation
};
struct InputResult { InputError error; std::string message; };
struct Event {
  bool reset;
  int reset_slot;
  std::vector<int> slots, tokens, positions;
};
struct CheckpointSlot { int sequence; int tokens_before; int tokens_after; };
struct Checkpoint { size_t event_index; std::vector<CheckpointSlot> slots; };
struct History {
  std::vector<Event> events;
  std::vector<Checkpoint> checkpoints;
  int max_rows;
};
inline constexpr size_t MAX_HISTORY_EVENTS = 1000000;
inline constexpr size_t MAX_HISTORY_ROWS = 4000000;
inline constexpr size_t MAX_HISTORY_CHECKPOINT_SLOTS = 4000000;
inline constexpr int MAX_EVENT_ROWS = 520;
inline constexpr size_t MAX_EVENT_LINE_BYTES = 32768;

namespace input_detail {
struct Line {
  std::string_view remaining;
  static bool space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f'; }
  std::string_view word() {
    while (!remaining.empty() && space(remaining.front())) remaining.remove_prefix(1);
    size_t n = 0;
    while (n < remaining.size() && !space(remaining[n])) ++n;
    auto value = remaining.substr(0, n);
    remaining.remove_prefix(n);
    return value;
  }
  bool integer(int& value) {
    const auto text = word();
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
  }
};
inline InputResult failure(InputError error, size_t line, const char* message) {
  return {error, "event " + std::to_string(line) + ": " + message};
}
}

// Checkpoints are 1-based event indices (RESET included), observing after-event state.
inline InputResult read_history(const std::string& path, int slot_count, int context,
    const std::vector<int>& checkpoint_event_indices, History& out) {
  if (path.empty() || slot_count <= 0 || context <= 0)
    return {InputError::BadArgument, "path, slots and context must be set and positive"};
  if (static_cast<size_t>(slot_count) > MAX_HISTORY_EVENTS ||
      checkpoint_event_indices.size() > MAX_HISTORY_EVENTS ||
      checkpoint_event_indices.size() > MAX_HISTORY_CHECKPOINT_SLOTS / static_cast<size_t>(slot_count))
    return {InputError::LimitExceeded, "slot/checkpoint metadata exceeds bounded history capacity"};
  int previous_checkpoint = 0;
  for (int index : checkpoint_event_indices) {
    if (index <= previous_checkpoint)
      return {InputError::BadArgument, "checkpoint event indices must be positive and strictly increasing"};
    if (static_cast<size_t>(index) > MAX_HISTORY_EVENTS)
      return {InputError::LimitExceeded, "checkpoint index exceeds event capacity"};
    previous_checkpoint = index;
  }
  try {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) return {InputError::Io, "cannot open event file: " + path};
    History history{{}, {}, 0};
    std::vector<int> counts(slot_count, 0), sequences(slot_count, -1);
    // Per-event marks avoid clearing slot_count entries on every small decode STEP.
    std::vector<size_t> seen(slot_count, 0);
    size_t rows_total = 0, checkpoint_index = 0;
    char buffer[MAX_EVENT_LINE_BYTES + 1];
    for (;;) {
      input.getline(buffer, sizeof(buffer));
      if (input.bad()) return {InputError::Io, "cannot read event file: " + path};
      if (input.fail() && !input.eof())
        return input_detail::failure(InputError::LimitExceeded, history.events.size() + 1, "event line exceeds byte capacity");
      const auto bytes = input.gcount();
      if (input.eof() && bytes == 0) break;
      const size_t event_index = history.events.size() + 1;
      if (event_index > MAX_HISTORY_EVENTS)
        return input_detail::failure(InputError::LimitExceeded, event_index, "too many events");
      input_detail::Line line{std::string_view(buffer, static_cast<size_t>(bytes) - (input.eof() ? 0 : 1))};
      const auto op = line.word();
      if (op.empty()) return input_detail::failure(InputError::MalformedEvent, event_index, "empty event line");
      if (op != "RESET" && op != "STEP")
        return input_detail::failure(InputError::UnsupportedEvent, event_index, "only ordinary RESET and STEP events are supported");
      const bool requested = checkpoint_index < checkpoint_event_indices.size() &&
          static_cast<size_t>(checkpoint_event_indices[checkpoint_index]) == event_index;
      std::vector<int> before;
      if (requested) before = counts;
      Event event{op == "RESET", -1, {}, {}, {}};
      if (event.reset) {
        if (!line.integer(event.reset_slot) || !line.word().empty())
          return input_detail::failure(InputError::MalformedEvent, event_index, "RESET requires exactly one integer slot");
        if (event.reset_slot < 0 || event.reset_slot >= slot_count)
          return input_detail::failure(InputError::SlotRange, event_index, "RESET slot out of range");
        counts[event.reset_slot] = 0;
        ++sequences[event.reset_slot];
      } else {
        int rows;
        if (!line.integer(rows)) return input_detail::failure(InputError::MalformedEvent, event_index, "STEP requires an integer row count");
        if (rows < 1 || rows > MAX_EVENT_ROWS)
          return input_detail::failure(InputError::LimitExceeded, event_index, "STEP rows must be in 1..520");
        if (rows_total + static_cast<size_t>(rows) > MAX_HISTORY_ROWS)
          return input_detail::failure(InputError::LimitExceeded, event_index, "too many total STEP rows");
        event.slots.reserve(rows); event.tokens.reserve(rows); event.positions.reserve(rows);
        int previous_slot = -1;
        for (int row = 0; row < rows; ++row) {
          int slot, token, position;
          if (!line.integer(slot) || !line.integer(token) || !line.integer(position))
            return input_detail::failure(InputError::MalformedEvent, event_index, "STEP row requires integer slot/token/position triples");
          if (slot < 0 || slot >= slot_count)
            return input_detail::failure(InputError::SlotRange, event_index, "STEP slot out of range");
          if (token < 0 || token >= VOCAB)
            return input_detail::failure(InputError::TokenRange, event_index, "STEP token outside model vocabulary");
          if (position < 0 || position >= context)
            return input_detail::failure(InputError::PositionRange, event_index, "STEP position outside context");
          if (sequences[slot] < 0)
            return input_detail::failure(InputError::MissingReset, event_index, "STEP slot must be explicitly RESET first");
          if (slot != previous_slot && seen[slot] == event_index)
            return input_detail::failure(InputError::NoncontiguousSlot, event_index, "STEP rows for a slot must be contiguous");
          if (position != counts[slot])
            return input_detail::failure(InputError::NonsequentialPosition, event_index, "STEP positions must advance sequentially from zero after RESET");
          seen[slot] = event_index;
          previous_slot = slot;
          ++counts[slot];
          event.slots.push_back(slot); event.tokens.push_back(token); event.positions.push_back(position);
        }
        if (!line.word().empty())
          return input_detail::failure(InputError::MalformedEvent, event_index, "extra tokens after STEP rows");
        rows_total += rows;
        history.max_rows = std::max(history.max_rows, rows);
      }
      history.events.push_back(std::move(event));
      if (requested) {
        Checkpoint checkpoint{event_index, {}};
        checkpoint.slots.reserve(slot_count);
        for (int slot = 0; slot < slot_count; ++slot) {
          if (sequences[slot] < 0)
            return input_detail::failure(InputError::MissingReset, event_index, "all slots must be explicitly RESET by a checkpoint");
          checkpoint.slots.push_back({sequences[slot], before[slot], counts[slot]});
        }
        history.checkpoints.push_back(std::move(checkpoint));
        ++checkpoint_index;
      }
    }
    if (checkpoint_index != checkpoint_event_indices.size())
      return {InputError::MissingCheckpoint, "requested checkpoint is beyond the recorded events"};
    if (history.events.empty()) return {InputError::MalformedEvent, "event history is empty"};
    out = std::move(history);
    return {InputError::None, {}};
  } catch (const std::bad_alloc&) {
    return {InputError::HostAllocation, "cannot allocate bounded event history"};
  } catch (const std::length_error&) {
    return {InputError::HostAllocation, "event history exceeds host container capacity"};
  } catch (const std::ios_base::failure&) {
    return {InputError::Io, "event file IO failed"};
  }
}
}
