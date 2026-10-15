#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace bonsai14 {
enum class Kind { gdn, attention };
struct Slot { Kind kind; unsigned index; };
enum class Error {
  shape, owner, boundary, manifest, busy, absent, order, capacity, handle,
  allocation, map, launch, status, copy, observer, drain, fence, abort,
  retirement, unready, uncertain, host_allocation, clock_exhausted
};
template<class T> using Result = std::variant<T, Error>;
struct Ok {};
using Status = Result<Ok>;
enum class Outcome { success, failure, uncertain };
enum class Operation { allocation, map, launch, status, copy, observer, drain };
enum class Stage { pending, allocated, mapped, launched, status_checked, copied, observed, drained };
enum class HandleKind { reservation, allocation, mapping, device, host, stream, event };
struct NotAcquired {};
struct Acquired { std::uint64_t native; };
struct Uncertain { std::optional<std::uint64_t> native; };
using Acquisition = std::variant<NotAcquired, Acquired, Uncertain>;
struct Ticket { std::size_t index; std::uint64_t generation; };
struct Owner { unsigned layer; std::uint64_t generation, boundary; };
struct ByteRange { std::size_t offset, bytes, allocation_bytes; };
struct ValidationHandle { Ticket ticket; Owner owner; ByteRange range; HandleKind kind; std::uint64_t native; };
struct ValidationRecord { Ticket ticket; Owner owner; ByteRange range; std::optional<std::uint64_t> native; };
struct LayerInput { Owner owner; std::vector<std::uint8_t> state, source; };
using Layers = std::array<LayerInput, 64>;
struct Reader { std::uint64_t id; };
using Manifest = std::array<std::vector<Reader>, 64>;

inline Result<Slot> own(unsigned layer) {
  if (layer >= 64) return Error::owner;
  if (layer % 4 == 3) return Slot{Kind::attention, layer / 4};
  return Slot{Kind::gdn, layer - layer / 4};
}
inline Result<unsigned> original(Slot s) {
  switch (s.kind) {
    case Kind::gdn:
      if (s.index >= 48) return Error::owner;
      return s.index / 3 * 4 + s.index % 3;
    case Kind::attention:
      if (s.index >= 16) return Error::owner;
      return s.index * 4 + 3;
  }
  return Error::owner;
}

class Generation {
  std::uint64_t generation_, boundary_;
  Layers layers_;
  Generation(std::uint64_t generation, std::uint64_t boundary, const Layers& layers)
    : generation_(generation), boundary_(boundary), layers_(layers) {}
public:
  static Result<std::shared_ptr<const Generation>> freeze(
      std::uint64_t generation, std::uint64_t boundary, const Layers& layers) {
    if (generation == 0) return Error::owner;
    for (unsigned l = 0; l < 64; ++l) {
      if (layers[l].owner.layer != l || layers[l].owner.generation != generation)
        return Error::owner;
      if (layers[l].owner.boundary != boundary) return Error::boundary;
      if (layers[l].state.empty() || layers[l].source.empty()) return Error::shape;
    }
    try {
      return std::shared_ptr<const Generation>(new Generation(generation, boundary, layers));
    } catch (const std::bad_alloc&) { return Error::host_allocation; }
      catch (const std::length_error&) { return Error::capacity; }
  }
  std::uint64_t generation() const { return generation_; }
  std::uint64_t boundary() const { return boundary_; }
  Result<const LayerInput*> lookup(unsigned l) const {
    if (l >= 64) return Error::owner;
    return &layers_[l];
  }
  const LayerInput& layer(unsigned l) const {
    if (l >= 64) std::terminate();
    return layers_[l];
  }
};
using Frozen = std::shared_ptr<const Generation>;
enum class HandleState { intent, live, uncertain_acquisition, uncertain_release, released, failed_release };
struct HandleView {
  HandleKind kind; HandleState state; std::optional<std::uint64_t> native;
};
struct InitialHandle { HandleKind kind; std::uint64_t native; };
struct Progress { Stage stage; std::optional<std::uint64_t> observed_at; };
struct ReaderProgress { Reader reader; Progress progress; };
enum class AttemptState { building, failed, fenced, aborting };

class Protocol {
  struct Custody { Frozen image; std::vector<HandleView> handles; };
  struct Attempt {
    Custody custody;
    std::array<Progress, 64> sources;
    std::array<std::vector<ReaderProgress>, 64> readers;
    std::size_t handle_capacity;
    AttemptState state;
    std::optional<Error> fault;
    std::optional<std::uint64_t> fenced_at;
    bool abort_fenced;
    std::uint64_t generation, boundary;
    Layers completed;
    std::array<bool,64> sealed{};
    std::vector<ValidationRecord> validation;
  };
  Custody head_;
  std::optional<Custody> retired_;
  std::optional<Attempt> attempt_;
  std::uint64_t clock_;

  Protocol(Custody head) : head_(std::move(head)), clock_(0) {}
  static bool valid_kind(HandleKind kind) {
    switch (kind) {
      case HandleKind::reservation: case HandleKind::allocation: case HandleKind::mapping:
      case HandleKind::device: case HandleKind::host: case HandleKind::stream:
      case HandleKind::event: return true;
    }
    return false;
  }
  static Error error(Operation op) {
    switch (op) {
      case Operation::allocation: return Error::allocation;
      case Operation::map: return Error::map;
      case Operation::launch: return Error::launch;
      case Operation::status: return Error::status;
      case Operation::copy: return Error::copy;
      case Operation::observer: return Error::observer;
      case Operation::drain: return Error::drain;
    }
    return Error::order;
  }
  static std::optional<std::pair<Stage, Stage>> edge(Operation op) {
    switch (op) {
      case Operation::allocation: return std::pair{Stage::pending, Stage::allocated};
      case Operation::map: return std::pair{Stage::allocated, Stage::mapped};
      case Operation::launch: return std::pair{Stage::mapped, Stage::launched};
      case Operation::status: return std::pair{Stage::launched, Stage::status_checked};
      case Operation::copy: return std::pair{Stage::status_checked, Stage::copied};
      case Operation::observer: return std::pair{Stage::copied, Stage::observed};
      case Operation::drain: return std::pair{Stage::observed, Stage::drained};
    }
    return std::nullopt;
  }
  bool aliases(HandleKind kind, std::uint64_t native, const HandleView* self) const {
    auto has = [&](const Custody& c) {
      for (const auto& h : c.handles)
        if (&h != self && h.state != HandleState::released && h.kind == kind &&
            h.native && *h.native == native) return true;
      return false;
    };
    return has(head_) || (retired_ && has(*retired_)) ||
      (attempt_ && has(attempt_->custody));
  }
  Status fail(Error e) {
    attempt_->state = AttemptState::failed;
    if (!attempt_->fault) attempt_->fault = e;
    return e;
  }
  Status step(Progress& p, Operation op, Outcome outcome) {
    auto transition = edge(op);
    if (!transition || p.stage != transition->first) return Error::order;
    if (outcome != Outcome::success) {
      if (outcome == Outcome::uncertain) return fail(Error::uncertain);
      if (outcome == Outcome::failure) return fail(error(op));
      return Error::order;
    }
    if (clock_ == std::numeric_limits<std::uint64_t>::max()) return fail(Error::clock_exhausted);
    ++clock_;
    p.stage = transition->second;
    if (op == Operation::observer) p.observed_at = clock_;
    return Ok{};
  }
  Status event_owner(Owner owner) const {
    if (!attempt_) return Error::absent;
    if (owner.layer >= 64 || owner.generation != attempt_->generation)
      return Error::owner;
    if (owner.boundary != attempt_->boundary) return Error::boundary;
    return Ok{};
  }
  static bool same_domain(const ValidationRecord& r, const ValidationHandle& v) {
    return r.ticket.index == v.ticket.index && r.ticket.generation == v.ticket.generation &&
      r.owner.layer == v.owner.layer && r.owner.generation == v.owner.generation &&
      r.owner.boundary == v.owner.boundary && r.range.offset == v.range.offset &&
      r.range.bytes == v.range.bytes && r.range.allocation_bytes == v.range.allocation_bytes &&
      r.native && *r.native == v.native;
  }
  void remember_validation(Ticket ticket, const HandleView& h) {
    for (auto& r : attempt_->validation) if (r.ticket.index == ticket.index) r.native = h.native;
  }
  Status validation_identity(const ValidationHandle& v) const {
    if (!attempt_) return Error::absent;
    if (v.ticket.generation != attempt_->generation || v.ticket.index >= attempt_->custody.handles.size()) return Error::handle;
    const auto& h = attempt_->custody.handles[v.ticket.index];
    if (h.kind != v.kind || (h.native && *h.native != v.native)) return Error::handle;
    for (const auto& r : attempt_->validation) if (same_domain(r, v)) return Ok{};
    return Error::handle;
  }
  Status validation_dead(Owner owner) const {
    if (auto s = event_owner(owner); std::holds_alternative<Error>(s)) return s;
    const unsigned l = owner.layer;
    if (!attempt_->sealed[l] || attempt_->completed[l].state.empty() || attempt_->completed[l].source.empty()) return Error::unready;
    if (attempt_->sources[l].stage != Stage::drained) return Error::unready;
    for (const auto& r : attempt_->readers[l]) if (r.progress.stage != Stage::drained) return Error::unready;
    return Ok{};
  }
  Status building() const {
    if (!attempt_) return Error::absent;
    if (attempt_->state != AttemptState::building) return Error::busy;
    return Ok{};
  }
  bool ready() const {
    if (!attempt_ || !attempt_->custody.image) return false;
    for (unsigned l = 0; l < 64; ++l) {
      if (attempt_->sources[l].stage != Stage::drained) return false;
      for (const auto& r : attempt_->readers[l])
        if (r.progress.stage != Stage::drained) return false;
    }
    for (const auto& h : attempt_->custody.handles)
      if (h.state != HandleState::live && h.state != HandleState::released) return false;
    return true;
  }
  template<class Release> static Status cleanup(Custody& c, Release&& release, Error failure) {
    for (auto it = c.handles.rbegin(); it != c.handles.rend(); ++it) {
      if (it->state == HandleState::released) continue;
      if (it->state != HandleState::live || !it->native) return Error::uncertain;
      Outcome result;
      try { result = release(it->kind, *it->native); }
      catch (...) { result = Outcome::uncertain; }
      switch (result) {
        case Outcome::success: it->state = HandleState::released; it->native.reset(); break;
        case Outcome::failure: return failure;
        case Outcome::uncertain: it->state = HandleState::uncertain_release; return Error::uncertain;
        default: it->state = HandleState::uncertain_release; return Error::uncertain;
      }
    }
    return Ok{};
  }
  static Status reconcile_release(Custody& c, std::size_t index, Outcome result) {
    if (index >= c.handles.size()) return Error::handle;
    auto& h = c.handles[index];
    if (h.state != HandleState::uncertain_release && h.state != HandleState::failed_release) return Error::order;
    switch (result) {
      case Outcome::success: h.state = HandleState::released; h.native.reset(); return Ok{};
      case Outcome::failure: h.state = HandleState::live; return Ok{};
      case Outcome::uncertain: return Error::uncertain;
    }
    return Error::order;
  }
public:
  Protocol(const Protocol&) = delete;
  Protocol& operator=(const Protocol&) = delete;
  Protocol(Protocol&& other) noexcept
    : head_(std::move(other.head_)), retired_(std::move(other.retired_)),
      attempt_(std::move(other.attempt_)), clock_(other.clock_) {
    other.head_.handles.clear();
    other.retired_.reset();
    other.attempt_.reset();
    other.clock_ = 0;
  }
  Protocol& operator=(Protocol&&) = delete;

  static Result<Protocol> bootstrap(Frozen image, std::span<const InitialHandle> handles) {
    if (!image) return Error::absent;
    try {
      Custody c{std::move(image), {}};
      c.handles.reserve(handles.size());
      for (auto h : handles) {
        if (h.native == 0 || !valid_kind(h.kind)) return Error::handle;
        for (const auto& prior : c.handles)
          if (prior.kind == h.kind && prior.native == h.native) return Error::handle;
        c.handles.push_back({h.kind, HandleState::live, h.native});
      }
      return Protocol(std::move(c));
    } catch (const std::bad_alloc&) { return Error::host_allocation; }
      catch (const std::length_error&) { return Error::capacity; }
  }
  const Frozen& head() const { return head_.image; }
  const Frozen* retained_old() const { return retired_ ? &retired_->image : nullptr; }
  const Frozen* candidate() const { return attempt_ ? &attempt_->custody.image : nullptr; }
  std::span<const HandleView> head_handles() const { return head_.handles; }
  std::span<const HandleView> old_handles() const {
    return retired_ ? std::span<const HandleView>(retired_->handles) : std::span<const HandleView>{};
  }
  std::span<const HandleView> candidate_handles() const {
    return attempt_ ? std::span<const HandleView>(attempt_->custody.handles) : std::span<const HandleView>{};
  }
  Status begin(Frozen image, const Manifest& manifest, std::size_t handle_capacity) {
    if (attempt_ || retired_) return Error::busy;
    if (!image || !head_.image) return Error::absent;
    if (head_.image->generation() == std::numeric_limits<std::uint64_t>::max() ||
        image->generation() != head_.image->generation() + 1) return Error::owner;
    if (image->boundary() <= head_.image->boundary()) return Error::boundary;
    if (handle_capacity == 0) return Error::capacity;
    for (const auto& readers : manifest) {
      if (readers.empty()) return Error::manifest;
      for (std::size_t i = 0; i < readers.size(); ++i) {
        if (readers[i].id == 0) return Error::manifest;
        for (std::size_t j = 0; j < i; ++j)
          if (readers[i].id == readers[j].id) return Error::manifest;
      }
    }
    try {
      Attempt a{{std::move(image), {}}, {}, {}, handle_capacity,
                AttemptState::building, std::nullopt, std::nullopt, false, 0, 0, {}, {}, {}};
      a.generation=a.custody.image->generation();a.boundary=a.custody.image->boundary();
      a.custody.handles.reserve(handle_capacity);
      a.validation.reserve(handle_capacity);
      for (unsigned l = 0; l < 64; ++l) {
        a.sources[l] = {Stage::pending, std::nullopt};
        a.readers[l].reserve(manifest[l].size());
        for (auto r : manifest[l]) a.readers[l].push_back({r, {Stage::pending, std::nullopt}});
      }
      attempt_.emplace(std::move(a));
      return Ok{};
    } catch (const std::bad_alloc&) { return Error::host_allocation; }
      catch (const std::length_error&) { return Error::capacity; }
  }
  Status begin_incremental(std::uint64_t generation,std::uint64_t boundary,const Manifest& manifest,std::size_t capacity) {
    if (attempt_ || retired_) return Error::busy;
    if (!head_.image) return Error::absent;
    if (head_.image->generation()==UINT64_MAX || generation!=head_.image->generation()+1) return Error::owner;
    if (boundary<=head_.image->boundary()) return Error::boundary;
    if (!capacity) return Error::capacity;
    for (const auto& rs:manifest) {
      if(rs.empty())return Error::manifest;
      for(size_t i=0;i<rs.size();i++){if(!rs[i].id)return Error::manifest;for(size_t j=0;j<i;j++)if(rs[j].id==rs[i].id)return Error::manifest;}
    }
    try {
      Attempt a{{nullptr,{}},{},{},capacity,AttemptState::building,std::nullopt,std::nullopt,false,generation,boundary,{}, {}, {}};
      a.custody.handles.reserve(capacity);
      a.validation.reserve(capacity);
      for(unsigned l=0;l<64;l++){a.sources[l]={Stage::pending,std::nullopt};a.readers[l].reserve(manifest[l].size());for(auto r:manifest[l])a.readers[l].push_back({r,{Stage::pending,std::nullopt}});}
      attempt_.emplace(std::move(a));return Ok{};
    }catch(const std::bad_alloc&){return Error::host_allocation;}catch(const std::length_error&){return Error::capacity;}
  }
  Status complete_layer(const LayerInput& image) {
    if(auto s=building();std::holds_alternative<Error>(s))return s;
    if(auto s=event_owner(image.owner);std::holds_alternative<Error>(s))return s;
    unsigned l=image.owner.layer;
    if(attempt_->sealed[l] || (l && !attempt_->sealed[l-1]))return Error::order;
    if(image.source.empty()||image.state.empty())return Error::shape;
    if(attempt_->sources[l].stage!=Stage::drained)return Error::unready;
    for(auto& r:attempt_->readers[l])if(r.progress.stage!=Stage::drained)return Error::unready;
    try{attempt_->completed[l]=image;attempt_->sealed[l]=true;return Ok{};}catch(const std::bad_alloc&){return fail(Error::host_allocation);}
  }
  Status freeze_completed() {
    if(auto s=building();std::holds_alternative<Error>(s))return s;
    for(bool sealed:attempt_->sealed)if(!sealed)return Error::unready;
    auto frozen=Generation::freeze(attempt_->generation,attempt_->boundary,attempt_->completed);
    if(auto e=std::get_if<Error>(&frozen))return fail(*e);
    attempt_->custody.image=std::get<Frozen>(std::move(frozen));return Ok{};
  }
  Status effect_failure(Error error) {
    if(auto s=building();std::holds_alternative<Error>(s))return s;
    return fail(error);
  }
  Result<Ticket> prepare_handle(HandleKind kind) {
    if (!valid_kind(kind)) return Error::handle;
    if (auto s = building(); std::holds_alternative<Error>(s)) return std::get<Error>(s);
    auto& a = *attempt_;
    if (a.custody.handles.size() == a.handle_capacity) return Error::capacity;
    const auto index = a.custody.handles.size();
    a.custody.handles.push_back({kind, HandleState::intent, std::nullopt});
    return Ticket{index, a.generation};
  }
  Result<Ticket> prepare_validation_handle(Owner owner, ByteRange range) {
    if (auto s = building(); std::holds_alternative<Error>(s)) return std::get<Error>(s);
    if (auto s = event_owner(owner); std::holds_alternative<Error>(s)) return std::get<Error>(s);
    if (attempt_->sealed[owner.layer]) return Error::order;
    // The backend releases whole allocations, never an interior byte slice.
    if (range.offset || !range.bytes || range.bytes != range.allocation_bytes) return Error::shape;
    auto prepared = prepare_handle(HandleKind::allocation);
    if (auto e = std::get_if<Error>(&prepared)) return *e;
    auto ticket = std::get<Ticket>(prepared);
    attempt_->validation.push_back({ticket, owner, range, std::nullopt});
    return ticket;
  }
  std::span<const ValidationRecord> validation_records() const {
    return attempt_ ? std::span<const ValidationRecord>(attempt_->validation) : std::span<const ValidationRecord>{};
  }
  Result<ValidationHandle> validation_handle(Ticket ticket) const {
    if (!attempt_ || ticket.generation != attempt_->generation) return Error::handle;
    for (const auto& r : attempt_->validation) if (r.ticket.index == ticket.index) {
      const auto& h = attempt_->custody.handles[ticket.index];
      if (!r.native || h.state != HandleState::live || h.native != r.native) return Error::uncertain;
      return ValidationHandle{r.ticket, r.owner, r.range, h.kind, *r.native};
    }
    return Error::handle;
  }
  template<class Release> Result<bool> release_validation_bounded(std::span<const ValidationHandle> handles, Release release, std::size_t limit) {
    if (auto s = building(); std::holds_alternative<Error>(s)) return std::get<Error>(s);
    if (!limit || limit > 256) return Error::capacity;
    for (std::size_t i = 0; i < handles.size(); ++i) {
      if (auto s = validation_identity(handles[i]); std::holds_alternative<Error>(s)) return std::get<Error>(s);
      if (auto s = validation_dead(handles[i].owner); std::holds_alternative<Error>(s)) return std::get<Error>(s);
      for (std::size_t j = 0; j < i; ++j) if (handles[j].ticket.index == handles[i].ticket.index) return Error::handle;
    }
    std::size_t count = 0;
    for (const auto& v : handles) {
      auto& h = attempt_->custody.handles[v.ticket.index];
      if (h.state == HandleState::released) continue;
      if (h.state == HandleState::failed_release) return Error::order;
      if (h.state != HandleState::live || !h.native) return Error::uncertain;
      if (count == limit) return false;
      ++count;
      Outcome outcome;
      try { outcome = release(v); } catch (...) { outcome = Outcome::uncertain; }
      switch (outcome) {
        case Outcome::success: h.state = HandleState::released; h.native.reset(); break;
        case Outcome::failure: h.state = HandleState::failed_release; return Error::retirement;
        case Outcome::uncertain: h.state = HandleState::uncertain_release; return Error::uncertain;
        default: h.state = HandleState::uncertain_release; return Error::uncertain;
      }
    }
    return true;
  }
  Status reconcile_validation_release(const ValidationHandle& handle, Outcome outcome) {
    if (auto s = validation_identity(handle); std::holds_alternative<Error>(s)) return s;
    return reconcile_release(attempt_->custody, handle.ticket.index, outcome);
  }
  Status resolve_handle(Ticket ticket, Acquisition acquired) {
    if (!attempt_) return Error::absent;
    auto& a = *attempt_;
    if (ticket.generation != a.generation ||
        ticket.index >= a.custody.handles.size()) return Error::handle;
    auto& h = a.custody.handles[ticket.index];
    if (h.state != HandleState::intent && h.state != HandleState::uncertain_acquisition)
      return Error::order;
    if (std::holds_alternative<NotAcquired>(acquired)) {
      if (h.state == HandleState::uncertain_acquisition && h.native) return Error::handle;
      h = {h.kind, HandleState::released, std::nullopt}; return Ok{};
    }
    if (auto p = std::get_if<Acquired>(&acquired)) {
      if (!p->native || (h.state==HandleState::uncertain_acquisition && h.native && *h.native!=p->native)) return Error::handle;
      if (aliases(h.kind, p->native, &h)) {
        h = {h.kind, HandleState::uncertain_acquisition, p->native};
        remember_validation(ticket, h);
        return fail(Error::handle);
      }
      h = {h.kind, HandleState::live, p->native}; remember_validation(ticket, h); return Ok{};
    }
    auto p = std::get<Uncertain>(acquired);
    if (p.native && *p.native == 0) return Error::handle;
    if(h.state==HandleState::uncertain_acquisition&&h.native){if(p.native&&p.native!=h.native)return Error::handle;p.native=h.native;}
    h = {h.kind, HandleState::uncertain_acquisition, p.native};
    remember_validation(ticket, h);
    return fail(Error::uncertain);
  }
  Status source(unsigned layer, Operation op, Outcome outcome) {
    if (auto s = building(); std::holds_alternative<Error>(s)) return s;
    if (layer >= 64) return Error::owner;
    return step(attempt_->sources[layer], op, outcome);
  }
  Status reader(unsigned layer, std::uint64_t id, Operation op, Outcome outcome) {
    if (auto s = building(); std::holds_alternative<Error>(s)) return s;
    if (layer >= 64) return Error::owner;
    for (auto& r : attempt_->readers[layer]) if (r.reader.id == id) {
      if (op == Operation::launch && attempt_->sources[layer].stage < Stage::copied)
        return Error::unready;
      return step(r.progress, op, outcome);
    }
    return Error::manifest;
  }
  Status source(Owner owner, Operation op, Outcome outcome) {
    if (auto s = event_owner(owner); std::holds_alternative<Error>(s)) return s;
    return source(owner.layer, op, outcome);
  }
  Status reader(Owner owner, std::uint64_t id, Operation op, Outcome outcome) {
    if (auto s = event_owner(owner); std::holds_alternative<Error>(s)) return s;
    return reader(owner.layer, id, op, outcome);
  }
  Status fence(Outcome outcome) {
    if (auto s = building(); std::holds_alternative<Error>(s)) return s;
    if (!ready()) return Error::unready;
    switch (outcome) {
      case Outcome::failure: return fail(Error::fence);
      case Outcome::uncertain: return fail(Error::uncertain);
      case Outcome::success: break;
      default: return Error::order;
    }
    if (clock_ == std::numeric_limits<std::uint64_t>::max()) return fail(Error::clock_exhausted);
    attempt_->fenced_at = ++clock_;
    attempt_->state = AttemptState::fenced;
    return Ok{};
  }
  Status publish() {
    if (!attempt_) return Error::absent;
    if (attempt_->fault) return *attempt_->fault;
    if (attempt_->state != AttemptState::fenced || !ready()) return Error::unready;
    if (retired_) return Error::busy;
    if (clock_ == std::numeric_limits<std::uint64_t>::max()) return fail(Error::clock_exhausted);
    ++clock_;
    retired_.emplace(std::move(head_));
    head_ = std::move(attempt_->custody);
    attempt_.reset();
    return Ok{};
  }
  Status abort_fence(Outcome outcome) {
    if (!attempt_) return Error::absent;
    attempt_->state = AttemptState::aborting;
    switch (outcome) {
      case Outcome::success: attempt_->abort_fenced = true; return Ok{};
      case Outcome::failure: attempt_->abort_fenced = false; return Error::abort;
      case Outcome::uncertain: attempt_->abort_fenced = false; return Error::uncertain;
    }
    return Error::order;
  }
  template<class Release> Status abort(Release&& release) {
    if (!attempt_) return Error::absent;
    if (!attempt_->abort_fenced) return Error::unready;
    auto result = cleanup(attempt_->custody, std::forward<Release>(release), Error::abort);
    if (std::holds_alternative<Ok>(result)) attempt_.reset();
    return result;
  }
  Status reconcile_abort_release(std::size_t index, Outcome result) {
    if (!attempt_) return Error::absent;
    return reconcile_release(attempt_->custody, index, result);
  }
  template<class Release> Status retire(Release&& release) {
    if (!retired_) return Error::absent;
    auto result = cleanup(*retired_, std::forward<Release>(release), Error::retirement);
    if (std::holds_alternative<Ok>(result)) retired_.reset();
    return result;
  }
  Status reconcile_retirement_release(std::size_t index, Outcome result) {
    if (!retired_) return Error::absent;
    return reconcile_release(*retired_, index, result);
  }
  template<class Release> Status close(Outcome fence_result, Release&& release) {
    if (attempt_ || retired_) return Error::busy;
    if (fence_result == Outcome::failure) return Error::fence;
    if (fence_result != Outcome::success) return Error::uncertain;
    auto result = cleanup(head_, std::forward<Release>(release), Error::retirement);
    if (std::holds_alternative<Ok>(result)) head_.image.reset();
    return result;
  }
  template<class Release> static Result<bool> cleanup_bounded(Custody& c, Release release, std::size_t limit) {
    if (!limit) return Error::capacity;
    std::size_t count=0;
    for(auto it=c.handles.rbegin();it!=c.handles.rend();++it) {
      if(it->state==HandleState::released)continue;
      if(count==limit)return false;
      if(it->state!=HandleState::live||!it->native)return Error::uncertain;
      ++count;auto outcome=release(it->kind,*it->native);
      if(outcome==Outcome::failure)return Error::retirement;
      if(outcome==Outcome::uncertain){it->state=HandleState::uncertain_release;return Error::uncertain;}
      if(outcome!=Outcome::success)return Error::order;
      it->state=HandleState::released;it->native.reset();
    }
    return true;
  }
  template<class Release> Result<bool> retire_bounded(Release release,std::size_t limit) {
    if(!retired_)return Error::absent;
    auto r=cleanup_bounded(*retired_,release,limit);
    if(auto done=std::get_if<bool>(&r);done&&*done)retired_.reset();return r;
  }
  template<class Release> Result<bool> abort_bounded(Outcome fence_result,Release release,std::size_t limit) {
    if(!attempt_)return Error::absent;
    auto s=abort_fence(fence_result);if(auto e=std::get_if<Error>(&s))return *e;
    auto r=cleanup_bounded(attempt_->custody,release,limit);
    if(auto done=std::get_if<bool>(&r);done&&*done)attempt_.reset();return r;
  }
  template<class Release> Result<bool> close_bounded(Outcome fence_result,Release release,std::size_t limit) {
    if(attempt_||retired_)return Error::busy;
    if(fence_result!=Outcome::success)return Error::uncertain;
    auto r=cleanup_bounded(head_,release,limit);
    if(auto done=std::get_if<bool>(&r);done&&*done)head_.image.reset();return r;
  }
  Status reconcile_head_release(std::size_t index, Outcome result) {
    return reconcile_release(head_, index, result);
  }
};
}
