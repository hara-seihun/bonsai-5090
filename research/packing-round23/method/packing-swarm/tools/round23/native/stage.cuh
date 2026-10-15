#pragma once
#include "session.cuh"
#include <fstream>
#include <memory>
#include <cstring>
#include "views.cuh"
namespace joined12 {
namespace rn=residual11::native;
struct Arena {
 std::vector<void*> handles;
 template<class T>T*make(size_t n,const T*src=nullptr){if(!n)return nullptr;try{handles.reserve(handles.size()+1);}catch(const std::bad_alloc&){return nullptr;}T*p=nullptr;if(binding15::allocate(&p,n*sizeof(T))!=cudaSuccess)return nullptr;handles.push_back(p);auto e=src?binding15::copy(p,src,n*sizeof(T),cudaMemcpyHostToDevice):binding15::fill(p,0xa5,n*sizeof(T));return e==cudaSuccess?p:nullptr;}
};
struct ValidationArena {
 std::optional<bonsai14::Owner> owner;std::vector<void*> handles;std::vector<bonsai14::ValidationHandle> validation;
 template<class T>T*make(size_t n,const T*src=nullptr){if(!n||!owner)return nullptr;handles.reserve(handles.size()+1);validation.reserve(validation.size()+1);T*p=nullptr;auto r=binding15::allocate_validation(&p,n*sizeof(T),*owner);auto*h=std::get_if<bonsai14::ValidationHandle>(&r);if(!h)return nullptr;handles.push_back(p);validation.push_back(*h);auto e=src?binding15::copy(p,src,n*sizeof(T),cudaMemcpyHostToDevice):binding15::fill(p,0xa5,n*sizeof(T));return e==cudaSuccess?p:nullptr;}
};
struct Raw {unsigned K,F;std::vector<uint8_t>raw,w;std::vector<uint16_t>s;};
inline bool load(const std::string&path,Raw&r){std::ifstream in(path,std::ios::binary);in.read((char*)&r.K,4);in.read((char*)&r.F,4);if(!in||!r.K||r.K%128||!r.F||r.F%32)return false;unsigned nkb=r.K/128,ng=r.F/32;r.raw.resize(uint64_t(r.F)*nkb*18);in.read((char*)r.raw.data(),r.raw.size());char more;if(!in||in.read(&more,1))return false;r.w.resize(uint64_t(nkb)*ng*512);r.s.resize(uint64_t(nkb)*r.F);
 for(unsigned kb=0;kb<nkb;kb++)for(unsigned g=0;g<ng;g++){
  for(unsigned tid=0;tid<32;tid++){unsigned row=g*32+16*((tid&3)>>1)+8*(tid&1)+tid/4;auto*p=r.raw.data()+(uint64_t(row)*nkb+kb)*18;r.s[(uint64_t(kb)*ng+g)*32+tid]=p[0]|uint16_t(p[1])<<8;}
  for(unsigned tid=0;tid<128;tid++){unsigned lane=tid/4,w=2*(tid&3),gid=lane>>2,tq=lane&3,row=g*32+16*(w>>2)+gid,h=(w>>1)&1;auto signs=[&](unsigned rr){auto*p=r.raw.data()+(uint64_t(rr)*nkb+kb)*18;unsigned b=2+8*h+tq;return unsigned(uint8_t(~p[b]))|(unsigned(uint8_t(~p[b+4]))<<8);};uint32_t word=signs(row)|(signs(row+8)<<16);std::memcpy(r.w.data()+(uint64_t(kb)*ng+g)*512+tid*4,&word,4);}
 }return true;}
inline bool f32load(const char*name,unsigned n,std::vector<float>&v){std::ifstream in(std::string("~/work/packing-mma-round12/archive/packing-swarm/round12/inputs/")+name+".f32",std::ios::binary);v.resize(n);in.read((char*)v.data(),n*4);char x;return bool(in)&&!in.read(&x,1);}
template<class T>bool read(std::vector<T>&v,const void*p,size_t n){v.resize(n);return binding15::copy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)==cudaSuccess;}
template<class T>bool same(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&(a.empty()||!std::memcmp(a.data(),b.data(),a.size()*sizeof(T)));}
struct NoBorrow {};
struct StridedFfn {views21::Borrowed gate,up,control_gate,control_up;};
struct StridedGdn {views21::Borrowed gate,control_gate;};
using BorrowRoute=std::variant<NoBorrow,StridedFfn,StridedGdn>;
struct Stage {
 BorrowRoute route=NoBorrow{};std::vector<float>xhost,x2host,gzhost;views21::Image material_x,material_x2,material_gz;rn::Observation materialized;
 bool borrowing()const{return !std::holds_alternative<NoBorrow>(route);}
 bool materialization_contract()const{auto exact=[](const views21::Image&a,const views21::Borrowed&b){return a&&b&&a->live()&&a->allocation.native==b->view->allocation.native&&a->allocation.ticket.index==b->view->allocation.ticket.index&&a->rows==b->view->rows&&a->stride==b->view->stride&&a->offset==b->view->offset&&a->columns==b->view->columns;};if(auto*f=std::get_if<StridedFfn>(&route))return exact(material_x,f->gate)&&exact(material_x2,f->up);if(auto*g=std::get_if<StridedGdn>(&route))return material_gz&&g->gate&&material_gz->live()&&material_gz->allocation.native==g->gate->view->allocation.native&&material_gz->rows==g->gate->view->rows&&material_gz->stride==16480&&material_gz->offset==0&&material_gz->columns==16384;return std::holds_alternative<NoBorrow>(route);}
 bool inputs(h5::PrepArgs&prep,std::vector<views21::Borrowed>&watch,bool control){
  if(auto*f=std::get_if<StridedFfn>(&route)){auto g=control?f->control_gate:f->gate;auto u=control?f->control_up:f->up;if(!g||!u||!g->live()||!u->live()||g->role!=views21::Role::gate||u->role!=views21::Role::up||g->source!=u->source||g->view->allocation.native!=u->view->allocation.native)return false;prep.x=g->data();prep.x2=u->data();prep.ldx=g->view->stride;watch={g,u};return true;}
  if(auto*g=std::get_if<StridedGdn>(&route)){auto v=control?g->control_gate:g->gate;if(!v||!v->live()||v->role!=views21::Role::gdn_gate)return false;prep.gz=v->data();prep.gz_ld=v->view->stride;watch={v};return true;}
  return std::holds_alternative<NoBorrow>(route);
 }
 ValidationArena arena;rn::Binding b{};rn::Session session;rn::Observation reference;std::vector<float>prior;bool observed=false;
 float*upload(std::span<const float>host,const views21::Image&view,unsigned B,unsigned F){if(!view)return arena.make(host.size(),host.data());if(host.size()!=size_t(B)*F)return nullptr;auto*p=arena.make<float>(host.size());if(!p||!std::holds_alternative<unsigned>(view->copy_to(p,B,F)))return nullptr;std::vector<float>image;if(!read(image,p,host.size())||std::memcmp(image.data(),host.data(),host.size_bytes()))return nullptr;return p;}
 rn::NativeResult<rn::Binding> open(residual11::Owner owner,unsigned B,unsigned flags,const Raw&raw,std::span<const float>x,std::span<const float>x2,std::span<const float>nw,std::span<const float>sign,std::span<const float>gz,std::span<const float>gnorm,std::span<const float>p,const views21::Image&vx,const views21::Image&vx2,const views21::Image&vgz,const BorrowRoute&borrow){
  route=borrow;if(std::holds_alternative<StridedFfn>(route)){xhost.assign(x.begin(),x.end());x2host.assign(x2.begin(),x2.end());}if(std::holds_alternative<StridedGdn>(route))gzhost.assign(gz.begin(),gz.end());material_x=vx;material_x2=vx2;material_gz=vgz;
  arena.owner=bonsai14::Owner{owner.layer,owner.generation,binding15::active->input()?binding15::active->input()->boundary:0};b.observation_id=binding15::active->projection();binding15::active->select(b.observation_id);unsigned n=raw.K,F=raw.F;auto adm=residual11::admit({n,B,B,n,flags|residual11::Store});if(!std::holds_alternative<residual11::Extents>(adm)||x.size()!=uint64_t(B)*n)return rn::NativeError::InvalidBinding;auto e=std::get<residual11::Extents>(adm);
  prior.assign(p.begin(),p.end());if(prior.size()!=uint64_t(B)*F)return rn::NativeError::InvalidBinding;b.owner=owner;b.batch=B;b.features=F;b.source_rows=B;b.prior=prior;b.prep.x=std::holds_alternative<StridedFfn>(route)?nullptr:upload(x,vx,B,n);b.xbytes=x.size_bytes();b.prep.ldx=n;b.prep.n=n;b.prep.flags=flags|residual11::Store;b.prep.ldo=n;
  if(flags&residual11::SiluMul){if(x2.size()!=x.size())return rn::NativeError::InvalidBinding;b.prep.x2=std::holds_alternative<StridedFfn>(route)?nullptr:upload(x2,vx2,B,n);b.x2bytes=x2.size_bytes();}
  if(flags&residual11::Norm){if(nw.size()!=n)return rn::NativeError::InvalidBinding;b.prep.norm_w=arena.make(n,nw.data());b.normbytes=n*4;}
  if(flags&(residual11::Sign|residual11::SignAfter)){if(sign.size()!=n)return rn::NativeError::InvalidBinding;b.prep.signs=arena.make(n,sign.data());b.signbytes=n*4;}
  if(flags&residual11::GdnNorm){if(gz.size()!=uint64_t(B)*16384||gnorm.size()!=128)return rn::NativeError::InvalidBinding;auto*parent=std::holds_alternative<StridedGdn>(route)?nullptr:upload(gz,vgz,B,16384);b.prep.gz=parent?parent+10240:nullptr;b.gzbytes=(gz.size()-10240)*4;b.prep.gz_ld=16384;b.prep.gnorm_w=arena.make(128,gnorm.data());b.gnormbytes=512;}
  if((std::holds_alternative<StridedFfn>(route)&&!(flags&residual11::SiluMul))||(std::holds_alternative<StridedGdn>(route)&&!(flags&residual11::GdnNorm))||!materialization_contract()||!inputs(b.prep,b.borrows,false))return rn::NativeError::InvalidBinding;
  for(auto&v:b.borrows){if(v->role==views21::Role::gate)b.xbytes=v->readable_bytes();else if(v->role==views21::Role::up)b.x2bytes=v->readable_bytes();else b.gzbytes=v->readable_bytes();}
  auto buf=[&](size_t bytes){return rn::Writable{arena.make<uint8_t>(bytes),bytes};};b.q=buf(e.q4);b.s=buf(e.s4);b.g=buf(e.g4);b.rq=buf(e.q4);b.rs=buf(e.s4);b.rg=buf(e.g4);b.transformed=buf(e.f32);b.norm=buf(e.norm);b.status=buf(B*(n/32)*4);b.rstatus=buf(B*(n/32)*4);b.prep.q4=(uint8_t*)b.q.data;b.prep.s4=(uint8_t*)b.s.data;b.prep.g4=(float*)b.g.data;b.prep.out_f32=(float*)b.transformed.data;b.prep.out_norm=(float*)b.norm.data;
  auto*w=arena.make(raw.w.size(),raw.w.data());auto*s=arena.make(raw.s.size(),raw.s.data());for(unsigned i=0;i<2;i++)b.readers.push_back({i+1,w,s,raw.w.size(),raw.s.size()*2,arena.make<float>(B*F),size_t(B)*F*4,arena.make<unsigned>(B*F),arena.make<unsigned>(B*F),size_t(B)*F*4});if(!std::holds_alternative<unsigned>(rn::validate(b)))return rn::NativeError::InvalidBinding;return b;
 }
 rn::NativeResult<rn::Observation> expected(bool copied){unsigned B=b.batch,n=b.prep.n,F=b.features;auto e=std::get<residual11::Extents>(residual11::admit({n,B,B,n,b.prep.flags}));h5::PrepArgs ref=b.prep;std::vector<views21::Borrowed>watch;
  if(borrowing()){if(copied){if(std::holds_alternative<StridedFfn>(route)){ref.x=upload(xhost,material_x,B,n);ref.x2=upload(x2host,material_x2,B,n);ref.ldx=n;if(!ref.x||!ref.x2)return rn::NativeError::Copy;}else{auto*parent=upload(gzhost,material_gz,B,16384);ref.gz=parent?parent+10240:nullptr;ref.gz_ld=16384;if(!parent)return rn::NativeError::Copy;}}else if(!inputs(ref,watch,true))return rn::NativeError::InvalidBinding;}
  auto live=[&](){for(auto&v:watch)if(!v->live()||v->source->identity.layer!=b.owner.layer||v->source->identity.generation!=b.owner.generation||v->view->rows!=B)return false;for(auto&v:b.borrows)if(!v->live())return false;return true;};
  ref.q4=arena.make<uint8_t>(e.q4);ref.s4=arena.make<uint8_t>(e.s4);ref.g4=arena.make<float>(B);ref.out_f32=arena.make<float>(B*n);ref.out_norm=arena.make<float>(B*n);if(!ref.q4||!ref.s4||!ref.g4||!ref.out_f32||!ref.out_norm)return rn::NativeError::Allocation;if(!live()||(!copied&&!binding15::ok(binding15::active->start(b.observation_id+6)))||!binding15::before(binding15::Effect::launch))return rn::NativeError::Launch;rn::k_prep2<<<B,n/32>>>(ref);if(binding15::finish(cudaGetLastError())!=cudaSuccess)return rn::NativeError::Launch;h5::PrepArgs rr{};rr.x=ref.out_f32;rr.ldx=n;rr.n=n;rr.flags=residual11::Quant|residual11::Residual;rr.q4=arena.make<uint8_t>(e.q4);rr.s4=arena.make<uint8_t>(e.s4);rr.g4=arena.make<float>(B);rr.ref_q4=ref.q4;rr.ref_s4=ref.s4;rr.ref_g4=ref.g4;if(!rr.q4||!rr.s4||!rr.g4)return rn::NativeError::Allocation;if(!live()||!binding15::before(binding15::Effect::launch))return rn::NativeError::Launch;rn::k_prep2<<<B,n/32>>>(rr);if(binding15::finish(cudaGetLastError())!=cudaSuccess)return rn::NativeError::Launch;
  auto*out=arena.make(prior.size(),prior.data());constexpr unsigned sm=rn::WsGeom<8,1,32,4>::SMEM;if(!out||!rr.q4||!rr.s4||!rr.g4||binding15::attribute(rn::k_gemm_fp4<8,1,32,1,4>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm)!=cudaSuccess)return rn::NativeError::Allocation;
  for(unsigned pass=0;pass<2;pass++){if(!live()||!binding15::before(binding15::Effect::launch))return rn::NativeError::Launch;rn::k_gemm_fp4<8,1,32,1,4><<<dim3((B+31)/32,(F/32+7)/8,1),288,sm>>>(pass?rr.q4:ref.q4,pass?rr.s4:ref.s4,pass?rr.g4:ref.g4,B,n,b.readers[0].w,(uint8_t*)b.readers[0].s,F,out,F,nullptr);if(binding15::finish(cudaGetLastError())!=cudaSuccess)return rn::NativeError::Launch;}
  if(!copied&&!binding15::ok(binding15::active->launched(b.observation_id+6)))return rn::NativeError::Launch;
  if(binding15::fence()!=cudaSuccess)return rn::NativeError::Fence;reference.owner=b.owner;
  if(!read(reference.q,ref.q4,e.q4)||!read(reference.s,ref.s4,e.s4)||!read(reference.g,ref.g4,B)||!read(reference.rq,rr.q4,e.q4)||!read(reference.rs,rr.s4,e.s4)||!read(reference.rg,rr.g4,B)||!read(reference.transformed,ref.out_f32,B*n)||!read(reference.norm,ref.out_norm,B*n))return rn::NativeError::Copy;reference.outputs.resize(2);if(!read(reference.outputs[0],out,B*F))return rn::NativeError::Copy;reference.outputs[1]=reference.outputs[0];return reference;
 }
 rn::NativeResult<std::vector<float>> run(){auto actual=session.run(b);if(!std::holds_alternative<rn::Observation*>(actual))return std::get<rn::NativeError>(actual);auto control=expected(false);if(!std::holds_alternative<rn::Observation>(control))return std::get<rn::NativeError>(control);auto&a=*std::get<rn::Observation*>(actual);auto&c=reference;
  if(!same(a.q,c.q)||!same(a.s,c.s)||!same(a.g,c.g)||!same(a.rq,c.rq)||!same(a.rs,c.rs)||!same(a.rg,c.rg)||!same(a.transformed,c.transformed)||!same(a.norm,c.norm)||a.outputs!=c.outputs)return rn::NativeError::ReferenceMismatch;
  if(borrowing()){auto strongest=reference;auto paid=expected(true);if(!std::holds_alternative<rn::Observation>(paid))return std::get<rn::NativeError>(paid);materialized=reference;reference=std::move(strongest);auto&m=materialized;if(!same(a.q,m.q)||!same(a.s,m.s)||!same(a.g,m.g)||!same(a.rq,m.rq)||!same(a.rs,m.rs)||!same(a.rg,m.rg)||!same(a.transformed,m.transformed)||!same(a.norm,m.norm)||a.outputs!=m.outputs)return rn::NativeError::ReferenceMismatch;}
  for(unsigned r=0;r<7;r++)if(!binding15::ok(binding15::active->observed(b.observation_id+r)))return rn::NativeError::Observe;
  std::vector<float>out(a.outputs[0].size());std::memcpy(out.data(),a.outputs[0].data(),out.size()*4);if(!std::holds_alternative<unsigned>(session.release()))return rn::NativeError::Release;observed=true;return out;
 }
};
}
