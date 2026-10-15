#pragma once
#include "generation.hpp"
#include <cstring>
#include <algorithm>
#include <cstdio>
namespace binding15 {
using namespace bonsai14;
enum class Effect { allocation, upload, fill, copy, observe, launch, attribute, fence, release, status, comparison, drain };
struct NativeAcquisition {Outcome outcome;Acquisition acquired;};
struct Event {Owner owner;uint64_t reader;Effect effect;Outcome outcome;std::optional<uint64_t> handle;std::optional<uint64_t> source;std::optional<size_t> bytes;std::optional<unsigned> variant;};
struct InputCustody {Frozen predecessor;uint64_t generation,boundary;std::vector<float> embedding;};
struct Fault {size_t event;Outcome outcome;bool after;};
class Binding {
 std::optional<Protocol> protocol_;std::vector<HandleView> reset_;std::vector<Event> events_;
 std::optional<InputCustody> input_;std::optional<Fault> fault_;size_t capacity_;
 Owner owner_{0,1,0};uint64_t reset_generation_=1,reset_boundary_=0,reader_=0;bool healthy_=true;unsigned projection_=0;unsigned rows_=1;
 std::array<std::vector<float>,64> outputs_;std::array<bool,64> begun_{};
 public:
 explicit Binding(size_t capacity):capacity_(capacity){events_.reserve(capacity);reset_.reserve(32768);}
 Binding(const Binding&)=delete;Binding&operator=(const Binding&)=delete;
 void fault(Fault f){fault_=f;}
 bool healthy()const{return healthy_;}
 void context(unsigned l,uint64_t generation,uint64_t boundary,uint64_t reader=0){owner_={l,generation,boundary};reader_=reader;}
 void select(uint64_t reader){reader_=reader;}
 std::span<const Event> events()const{return events_;}
 bool accept(Effect effect){return record_before(effect)&&finish(Outcome::success)==Outcome::success;}
 Status reject(Error error){healthy_=false;return protocol_&&input_?protocol_->effect_failure(error):Status{error};}
 void details(std::optional<uint64_t> source,std::optional<size_t> bytes,std::optional<unsigned> variant){if(events_.empty())std::terminate();events_.back().source=source;events_.back().bytes=bytes;events_.back().variant=variant;}
 template<class Backend> Status acquire(Backend backend){
   auto t=prepare();if(auto e=std::get_if<Error>(&t))return *e;auto ticket=std::get<Ticket>(t);
   if(!record_before(Effect::allocation)){resolve(ticket,NotAcquired{});return Error::allocation;}
   NativeAcquisition result=backend();if(auto p=std::get_if<Acquired>(&result.acquired))events_.back().handle=p->native;else if(auto p=std::get_if<Uncertain>(&result.acquired))events_.back().handle=p->native;
   auto outcome=finish(result.outcome);
   if(outcome!=Outcome::success)if(auto p=std::get_if<Acquired>(&result.acquired))result.acquired=Uncertain{p->native};
   auto s=resolve(ticket,result.acquired);if(std::holds_alternative<Error>(s))return s;return outcome==Outcome::success?Status{Ok{}}:Status{Error::uncertain};
 }
 template<class Backend> Result<ValidationHandle> acquire_validation(Owner owner,ByteRange range,Backend backend){
   if(!healthy_||!protocol_||!input_)return Error::busy;
   if(owner.layer!=owner_.layer||owner.generation!=owner_.generation||owner.boundary!=owner_.boundary)return Error::owner;
   auto t=protocol_->prepare_validation_handle(owner,range);if(auto e=std::get_if<Error>(&t))return *e;auto ticket=std::get<Ticket>(t);
   if(!record_before(Effect::allocation)){resolve(ticket,NotAcquired{});return Error::allocation;}
   NativeAcquisition result=[&](){try{return backend();}catch(...){return NativeAcquisition{Outcome::uncertain,Uncertain{std::nullopt}};}}();
   if(auto p=std::get_if<Acquired>(&result.acquired))events_.back().handle=p->native;else if(auto p=std::get_if<Uncertain>(&result.acquired))events_.back().handle=p->native;
   details(std::nullopt,range.bytes,std::nullopt);auto outcome=finish(result.outcome);
   if(outcome!=Outcome::success)if(auto p=std::get_if<Acquired>(&result.acquired))result.acquired=Uncertain{p->native};
   auto s=resolve(ticket,result.acquired);if(auto e=std::get_if<Error>(&s))return *e;
   if(outcome!=Outcome::success)return Error::uncertain;
   return protocol_->validation_handle(ticket);
 }
 template<class Release> Result<bool> release_validation_step(std::span<const ValidationHandle> handles,Release release){
   if(!protocol_||!input_)return Error::absent;
   return protocol_->release_validation_bounded(handles,[&](const ValidationHandle& v){
     auto saved_owner=owner_;auto saved_reader=reader_;owner_=v.owner;reader_=0;
     auto outcome=cleanup_effect(Effect::release,v.native,[&](){
       try{return release(v);}catch(...){return Outcome::uncertain;}
     });
     if(!events_.empty()&&events_.back().effect==Effect::release&&events_.back().handle==v.native)details(v.native,v.range.bytes,std::nullopt);
     owner_=saved_owner;reader_=saved_reader;return outcome;
   },256);
 }
 Status reconcile_validation_release(const ValidationHandle& handle,Outcome outcome){return protocol_?protocol_->reconcile_validation_release(handle,outcome):Status{Error::absent};}
 template<class Backend> Outcome effect(Effect effect,std::optional<uint64_t> handle,Backend backend){if(!record_before(effect,handle))return events_.empty()?Outcome::failure:events_.back().outcome;return finish(backend());}
 template<class Backend> Outcome cleanup_effect(Effect effect,std::optional<uint64_t> handle,Backend backend){
  if(events_.size()==capacity_)return Outcome::failure;events_.push_back({owner_,reader_,effect,Outcome::uncertain,handle,std::nullopt,std::nullopt,std::nullopt});
  if(fault_&&fault_->event==events_.size()-1&&!fault_->after){events_.back().outcome=fault_->outcome;return fault_->outcome;}
  auto result=backend();if(fault_&&fault_->event==events_.size()-1&&fault_->after)result=Outcome::uncertain;events_.back().outcome=result;return result;
 }
 Protocol* protocol(){return protocol_?&*protocol_:nullptr;}
 void dump(FILE*file)const{
  std::fprintf(file,"{\"state\":\"native_effect_custody\",\"healthy\":%s,\"event_count\":%zu,\"head_generation\":%llu,\"handles\":[",healthy_?"true":"false",events_.size(),(unsigned long long)(protocol_&&protocol_->head()?protocol_->head()->generation():0));bool comma=false;
  auto handles=[&](const char*scope,std::span<const HandleView>hs){for(size_t i=0;i<hs.size();i++){auto&h=hs[i];if(comma)std::fputc(',',file);comma=true;std::fprintf(file,"{\"scope\":\"%s\",\"ticket\":%zu,\"kind\":%u,\"state\":%u,\"native\":",scope,i,unsigned(h.kind),unsigned(h.state));if(h.native)std::fprintf(file,"%llu",(unsigned long long)*h.native);else std::fputs("null",file);std::fputc('}',file);}};
  handles("reset",reset_);if(protocol_){handles("head",protocol_->head_handles());handles("old",protocol_->old_handles());handles("candidate",protocol_->candidate_handles());}
  std::fputs("],\"validation_domains\":[",file);comma=false;if(protocol_)for(const auto&r:protocol_->validation_records()){if(comma)std::fputc(',',file);comma=true;std::fprintf(file,"{\"ticket\":%zu,\"generation\":%llu,\"boundary\":%llu,\"layer\":%u,\"offset\":%zu,\"bytes\":%zu,\"allocation_bytes\":%zu,\"native\":",r.ticket.index,(unsigned long long)r.owner.generation,(unsigned long long)r.owner.boundary,r.owner.layer,r.range.offset,r.range.bytes,r.range.allocation_bytes);if(r.native)std::fprintf(file,"%llu",(unsigned long long)*r.native);else std::fputs("null",file);std::fputc('}',file);}
  std::fputs("],\"events\":[",file);comma=false;for(size_t i=0;i<events_.size();i++){auto&e=events_[i];if(comma)std::fputc(',',file);comma=true;std::fprintf(file,"{\"event\":%zu,\"layer\":%u,\"generation\":%llu,\"boundary\":%llu,\"reader\":%llu,\"effect\":%u,\"outcome\":%u,\"handle\":",i,e.owner.layer,(unsigned long long)e.owner.generation,(unsigned long long)e.owner.boundary,(unsigned long long)e.reader,unsigned(e.effect),unsigned(e.outcome));if(e.handle)std::fprintf(file,"%llu",(unsigned long long)*e.handle);else std::fputs("null",file);std::fputs(",\"source\":",file);if(e.source)std::fprintf(file,"%llu",(unsigned long long)*e.source);else std::fputs("null",file);std::fputs(",\"bytes\":",file);if(e.bytes)std::fprintf(file,"%zu",*e.bytes);else std::fputs("null",file);std::fputs(",\"variant\":",file);if(e.variant)std::fprintf(file,"%u",*e.variant);else std::fputs("null",file);std::fputc('}',file);}std::fputs("]}\n",file);
 }
 std::span<const HandleView> reset_handles()const{return reset_;}
 uint64_t reset_generation()const{return reset_generation_;}uint64_t reset_boundary()const{return reset_boundary_;}
 bool reset_identity(uint64_t generation,uint64_t boundary){if(protocol_||input_||!reset_.empty()||!generation)return false;reset_generation_=generation;reset_boundary_=boundary;return true;}
 const InputCustody* input()const{return input_?&*input_:nullptr;}
 static Manifest manifest(){Manifest m;for(unsigned l=0;l<64;l++){m[l].push_back({1});unsigned projections=l==63?5:4;for(unsigned p=0;p<projections;p++)for(unsigned r=0;r<7;r++)m[l].push_back({2+p*7+r});if(l==63)m[l].push_back({37});for(unsigned id=38;id<46;id++)m[l].push_back({id});}return m;}
 bool record_before(Effect effect,std::optional<uint64_t> handle={}){
   if(!healthy_||events_.size()==capacity_){healthy_=false;return false;}
   events_.push_back({owner_,reader_,effect,Outcome::uncertain,handle,std::nullopt,std::nullopt,std::nullopt});
   if(fault_&&fault_->event==events_.size()-1&&!fault_->after){finish(fault_->outcome);return false;}return true;
 }
 Outcome finish(Outcome outcome){
   if(events_.empty())std::terminate();
   if(fault_&&fault_->event==events_.size()-1&&fault_->after)outcome=fault_->outcome;
   events_.back().outcome=outcome;
   if(outcome!=Outcome::success){healthy_=false;if(protocol_&&input_)protocol_->effect_failure(outcome==Outcome::uncertain?Error::uncertain:Error::launch);}
   return outcome;
 }
 Result<Ticket> prepare(){if(!healthy_)return Error::busy;if(protocol_&&input_)return protocol_->prepare_handle(HandleKind::allocation);
   if(reset_.size()==reset_.capacity())return Error::capacity;
   reset_.push_back({HandleKind::allocation,HandleState::intent,std::nullopt});return Ticket{reset_.size()-1,reset_generation_};}
 Status resolve(Ticket t,Acquisition a){if(protocol_&&input_)return protocol_->resolve_handle(t,a);
   if(t.generation!=reset_generation_||t.index>=reset_.size())return Error::handle;auto&h=reset_[t.index];
   if(h.state!=HandleState::intent&&h.state!=HandleState::uncertain_acquisition)return Error::order;
   if(auto p=std::get_if<Acquired>(&a)){if(!p->native||(h.state==HandleState::uncertain_acquisition&&h.native&&*h.native!=p->native))return Error::handle;h={h.kind,HandleState::live,p->native};return Ok{};}
   if(std::holds_alternative<NotAcquired>(a)){if(h.state==HandleState::uncertain_acquisition&&h.native)return Error::handle;h={h.kind,HandleState::released,{}};return Ok{};}
   auto p=std::get<Uncertain>(a);if(h.state==HandleState::uncertain_acquisition&&h.native){if(p.native&&p.native!=h.native)return Error::handle;p.native=h.native;}
   h={h.kind,HandleState::uncertain_acquisition,p.native};healthy_=false;return Error::uncertain;
 }
 template<class Release>Status reset_close(Outcome fence,Release release){
  if(protocol_||input_)return Error::busy;if(fence!=Outcome::success)return Error::uncertain;
  for(size_t i=reset_.size();i>0;i--){auto&h=reset_[i-1];if(h.state==HandleState::released)continue;if(h.state!=HandleState::live||!h.native)return Error::uncertain;
   auto outcome=cleanup_effect(Effect::release,h.native,[&](){return release(h.kind,*h.native);});if(outcome==Outcome::failure)return Error::retirement;if(outcome==Outcome::uncertain){h.state=HandleState::uncertain_release;return Error::uncertain;}h.state=HandleState::released;h.native.reset();
  }reset_.clear();return Ok{};
 }
 template<class Release>Result<bool> reset_close_step(Outcome fence,Release release){
  if(protocol_||input_)return Error::busy;if(fence!=Outcome::success)return Error::uncertain;size_t count=0;
  for(size_t i=reset_.size();i>0;i--){auto&h=reset_[i-1];if(h.state==HandleState::released)continue;if(count==256)return false;if(h.state!=HandleState::live||!h.native)return Error::uncertain;++count;
   auto outcome=cleanup_effect(Effect::release,h.native,[&](){return release(h.kind,*h.native);});if(outcome==Outcome::failure)return Error::retirement;if(outcome==Outcome::uncertain){h.state=HandleState::uncertain_release;return Error::uncertain;}h.state=HandleState::released;h.native.reset();
  }reset_.clear();return true;
 }
 Status reconcile_reset_release(size_t index,Outcome outcome){if(protocol_||index>=reset_.size())return Error::handle;auto&h=reset_[index];if(h.state!=HandleState::uncertain_release)return Error::order;if(outcome==Outcome::uncertain)return Error::uncertain;if(outcome==Outcome::success){h.state=HandleState::released;h.native.reset();}else h.state=HandleState::live;return Ok{};}
 Status bootstrap(const Layers& layers){if(protocol_||input_)return Error::busy;auto frozen=Generation::freeze(reset_generation_,reset_boundary_,layers);if(auto e=std::get_if<Error>(&frozen))return *e;std::vector<InitialHandle> handles;handles.reserve(reset_.size());for(auto&h:reset_){if(h.state!=HandleState::live||!h.native)return Error::uncertain;handles.push_back({h.kind,*h.native});}auto p=Protocol::bootstrap(std::get<Frozen>(frozen),handles);if(auto e=std::get_if<Error>(&p))return *e;protocol_.emplace(std::get<Protocol>(std::move(p)));reset_.clear();return Ok{};}
 Status begin(uint64_t generation,uint64_t boundary,std::span<const float> embedding,unsigned rows){if(!healthy_||!protocol_||input_)return Error::busy;if(!rows||rows>8||embedding.size()!=uint64_t(rows)*5120)return Error::shape;rows_=rows;
   InputCustody candidate{protocol_->head(),generation,boundary,{embedding.begin(),embedding.end()}};
   auto s=protocol_->begin_incremental(generation,boundary,manifest(),32768);if(std::holds_alternative<Error>(s))return s;
   input_.emplace(std::move(candidate));begun_.fill(false);for(auto&o:outputs_)o.clear();return Ok{};}
 Status layer(unsigned l,std::span<const float> activation){if(!input_||l>=64||begun_[l])return Error::owner;if((l&&!begun_[l-1])||activation.size()!=uint64_t(rows_)*5120)return Error::order;
   const auto&expected=l?outputs_[l-1]:input_->embedding;if(expected.size()!=activation.size()||std::memcmp(expected.data(),activation.data(),activation.size_bytes()))return Error::boundary;
   context(l,input_->generation,input_->boundary);projection_=0;begun_[l]=true;
   for(auto op:{Operation::allocation,Operation::map,Operation::launch,Operation::status,Operation::copy}){auto s=protocol_->source(owner_,op,Outcome::success);if(std::holds_alternative<Error>(s))return s;}
   return Ok{};
 }
 uint64_t projection(){return 2+7*projection_++;}
 Status start(uint64_t id){reader_=id;for(auto op:{Operation::allocation,Operation::map}){auto s=protocol_->reader(owner_,id,op,Outcome::success);if(std::holds_alternative<Error>(s))return s;}return Ok{};}
 Status launched(uint64_t id){reader_=id;return protocol_->reader(owner_,id,Operation::launch,Outcome::success);}
 Status observed(uint64_t id){reader_=id;for(auto op:{Operation::status,Operation::copy,Operation::observer,Operation::drain}){Effect effect=op==Operation::status?Effect::status:op==Operation::copy?Effect::copy:op==Operation::observer?Effect::comparison:Effect::drain;if(!accept(effect))return Error::uncertain;auto s=protocol_->reader(owner_,id,op,Outcome::success);if(std::holds_alternative<Error>(s))return s;}return Ok{};}
 Status complete(LayerInput image,std::span<const float> output){if(!input_||image.owner.layer!=owner_.layer||output.size()!=uint64_t(rows_)*5120)return Error::owner;
   if(auto s=start(1);std::holds_alternative<Error>(s))return s;
   if(auto s=launched(1);std::holds_alternative<Error>(s))return s;
   if(auto s=observed(1);std::holds_alternative<Error>(s))return s;
   for(auto op:{Operation::observer,Operation::drain}){auto s=protocol_->source(owner_,op,Outcome::success);if(std::holds_alternative<Error>(s))return s;}
   auto s=protocol_->complete_layer(image);if(std::holds_alternative<Error>(s))return s;outputs_[owner_.layer].assign(output.begin(),output.end());return Ok{};
 }
 Status freeze(){return protocol_&&input_?protocol_->freeze_completed():Status{Error::absent};}
 Status publish(Outcome fence){if(!healthy_||!protocol_||!input_)return Error::unready;auto s=protocol_->fence(fence);if(std::holds_alternative<Error>(s))return s;s=protocol_->publish();if(std::holds_alternative<Ok>(s))input_.reset();return s;}
 template<class Release>Status retire(Release release){if(!protocol_)return Error::absent;return protocol_->retire([&](HandleKind k,uint64_t h){return cleanup_effect(Effect::release,h,[&](){return release(k,h);});});}
 template<class Release>Status abort(Outcome fence,Release release){if(!protocol_||!input_)return Error::absent;auto s=protocol_->abort_fence(fence);if(std::holds_alternative<Error>(s))return s;s=protocol_->abort([&](HandleKind k,uint64_t h){return cleanup_effect(Effect::release,h,[&](){return release(k,h);});});if(std::holds_alternative<Ok>(s)){input_.reset();healthy_=true;}return s;}
 template<class Release>Result<bool> retire_step(Release release){if(!protocol_)return Error::absent;return protocol_->retire_bounded([&](HandleKind k,uint64_t h){return cleanup_effect(Effect::release,h,[&](){return release(k,h);});},256);}
 template<class Release>Result<bool> abort_step(Outcome fence,Release release){if(!protocol_||!input_)return Error::absent;auto r=protocol_->abort_bounded(fence,[&](HandleKind k,uint64_t h){return cleanup_effect(Effect::release,h,[&](){return release(k,h);});},256);if(auto done=std::get_if<bool>(&r);done&&*done){input_.reset();healthy_=true;}return r;}
 template<class Release>Result<bool> close_step(Outcome fence,Release release){if(!protocol_)return Error::absent;return protocol_->close_bounded(fence,[&](HandleKind k,uint64_t h){return cleanup_effect(Effect::release,h,[&](){return release(k,h);});},256);}
 template<class Release>Status close(Outcome fence,Release release){return protocol_?protocol_->close(fence,[&](HandleKind k,uint64_t h){return cleanup_effect(Effect::release,h,[&](){return release(k,h);});}):Status{Error::absent};}
};
inline bool ok(Status s){return std::holds_alternative<Ok>(s);}
}
