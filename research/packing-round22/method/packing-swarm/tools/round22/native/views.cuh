#pragma once
#include "effects.cuh"
#include <memory>
#include "../source.hpp"
namespace views21 {
enum class Error {shape,identity,lifetime,copy};
class RowView {
 RowView(bonsai14::ValidationHandle h,unsigned b,unsigned s,unsigned o,unsigned f):allocation(h),rows(b),stride(s),offset(o),columns(f){}
 public:
 const bonsai14::ValidationHandle allocation;const unsigned rows,stride,offset,columns;
 using Image=std::shared_ptr<const RowView>;
 static std::variant<Image,Error> make(const bonsai14::ValidationHandle&h,unsigned b,unsigned s,unsigned o,unsigned f){
  if((b!=1&&b!=8&&b!=32)||(s!=16480&&s!=14336&&s!=34816)||!f||o>s||f>s-o||h.range.offset||h.range.bytes!=h.range.allocation_bytes||h.range.bytes!=size_t(b)*s*4||h.kind!=bonsai14::HandleKind::allocation||!h.native)return Error::shape;
  Image v(new RowView(h,b,s,o,f));if(!v->live())return Error::identity;return v;
 }
 bool live()const{
  if(!binding15::active||!binding15::active->healthy())return false;auto*p=binding15::active->protocol();if(!p)return false;auto hs=p->candidate_handles();if(allocation.ticket.index>=hs.size())return false;auto&h=hs[allocation.ticket.index];
  if(h.state!=bonsai14::HandleState::live||!h.native||*h.native!=allocation.native||h.kind!=allocation.kind)return false;
  bool exact=false;for(auto&r:p->validation_records())if(r.ticket.index==allocation.ticket.index&&r.ticket.generation==allocation.ticket.generation&&r.owner.layer==allocation.owner.layer&&r.owner.generation==allocation.owner.generation&&r.owner.boundary==allocation.owner.boundary&&r.range.offset==allocation.range.offset&&r.range.bytes==allocation.range.bytes&&r.range.allocation_bytes==allocation.range.allocation_bytes&&r.native&&*r.native==allocation.native)exact=true;
  auto*e=binding15::evidence_record(allocation.native);return exact&&e&&e->state==binding15::Evidence::live&&e->owner.layer==allocation.owner.layer&&e->owner.generation==allocation.owner.generation&&e->owner.boundary==allocation.owner.boundary&&e->stream==binding15::current_stream&&e->domain==binding15::current_domain;
 }
 private:
 const float*row(unsigned r)const{return reinterpret_cast<const float*>(uintptr_t(allocation.native))+size_t(r)*stride+offset;}
 public:
 std::variant<unsigned,Error> copy_to(float*out,unsigned b,unsigned f)const{
  if(!out||b!=rows||f!=columns)return Error::shape;if(!live())return Error::lifetime;
  for(unsigned r=0;r<rows;r++)if(binding15::copy(out+size_t(r)*columns,row(r),size_t(columns)*4,cudaMemcpyDeviceToDevice)!=cudaSuccess)return Error::copy;
  return rows;
 }
};
using Image=RowView::Image;
enum class Role {gate,up,gdn_gate};
class Borrow {
 Borrow(views21::Image v,source19::Supplied s,Role r,uint64_t hash):view(std::move(v)),source(std::move(s)),role(r),output_hash(hash),stream(binding15::current_stream),domain(binding15::current_domain){}
 public:
 const views21::Image view;const source19::Supplied source;const Role role;const uint64_t output_hash;const unsigned stream;const std::array<unsigned,5>domain;
 using Image=std::shared_ptr<const Borrow>;
 static std::variant<Image,Error> make(const views21::Image&v,const source19::Supplied&s,Role r,uint64_t hash){
  if(!v||!s||!source19::intact(s)||!v->live()||v->rows!=s->identity.batch||v->allocation.owner.layer!=s->identity.layer||v->allocation.owner.generation!=s->identity.generation||v->allocation.owner.boundary!=s->identity.boundary||s->identity.stream!=binding15::current_stream)return Error::identity;
  bool shape=false;switch(r){case Role::gate:shape=s->identity.region==source19::Region::post_attention&&v->stride==34816&&v->offset==0&&v->columns==17408;break;case Role::up:shape=s->identity.region==source19::Region::post_attention&&v->stride==34816&&v->offset==17408&&v->columns==17408;break;case Role::gdn_gate:shape=s->identity.region==source19::Region::layer_input&&s->identity.layer%4!=3&&v->stride==16480&&v->offset==10240&&v->columns==6144;break;}
  if(!shape)return Error::shape;Image b(new Borrow(v,s,r,hash));if(!b->live())return Error::identity;return b;
 }
 size_t readable_bytes()const{return (size_t(view->rows-1)*view->stride+view->columns)*4;}
 const float*data()const{return reinterpret_cast<const float*>(uintptr_t(view->allocation.native))+view->offset;}
 bool live()const{if(!view->live()||!source19::intact(source)||stream!=binding15::current_stream||domain!=binding15::current_domain||!binding15::active->input())return false;auto&i=*binding15::active->input();return i.generation==source->identity.generation&&i.boundary==source->identity.boundary&&view->allocation.owner.layer==source->identity.layer&&source->family.token==source->identity.token&&source->identity.stream==stream;}
};
using Borrowed=Borrow::Image;
static_assert(sizeof(RowView)==96);static_assert(sizeof(Image)==16);
}
