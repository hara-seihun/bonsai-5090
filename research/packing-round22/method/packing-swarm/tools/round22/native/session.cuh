#pragma once
#include "~/work/packing-mma-round11/method/packing-swarm/tools/round11/residual/protocol.hpp"
#include "~/work/packing-mma-round11/method/packing-swarm/tools/round11/residual/literal.cuh"
#include "effects.cuh"
#include "views.cuh"
#include <vector>
#include <optional>
namespace residual11::native {
enum class NativeError { InvalidBinding, Busy, Allocation, Copy, Launch, Fence, Status, Observe, ReferenceMismatch, Release, MissingNextInputABConvPendingOwnership };
template<class T>using NativeResult=std::variant<T,NativeError>;
struct Span {const void*data;size_t bytes;};
struct Writable {void*data;size_t bytes;};
struct Reader {unsigned id;const uint8_t*w;const uint16_t*s;size_t wbytes,sbytes;float*out;size_t outbytes;unsigned *owners0,*owners1;size_t ownerbytes;};
struct Binding {uint64_t observation_id;Owner owner;h5::PrepArgs prep;unsigned batch,features,source_rows;size_t xbytes,x2bytes,normbytes,signbytes,gzbytes,gnormbytes;Writable q,s,g,rq,rs,rg,transformed,norm,status,rstatus;std::span<const float>prior;std::vector<Reader>readers;std::vector<views21::Borrowed>borrows;};
inline bool live_borrows(const Binding&b){for(auto&v:b.borrows){if(!v||!v->live()||v->source->identity.layer!=b.owner.layer||v->source->identity.generation!=b.owner.generation||v->view->rows!=b.batch)return false;switch(v->role){case views21::Role::gate:if(b.prep.x!=v->data()||b.prep.ldx!=int(v->view->stride)||b.xbytes!=v->readable_bytes())return false;break;case views21::Role::up:if(b.prep.x2!=v->data()||b.prep.ldx!=int(v->view->stride)||b.x2bytes!=v->readable_bytes())return false;break;case views21::Role::gdn_gate:if(b.prep.gz!=v->data()||b.prep.gz_ld!=int(v->view->stride)||b.gzbytes!=v->readable_bytes())return false;break;}}return true;}
struct Observation {Owner owner;std::vector<uint8_t>q,s,rq,rs;std::vector<float>g,rg,transformed,norm;std::vector<unsigned>status,rstatus;std::vector<std::vector<unsigned>>reader_status;std::vector<std::vector<uint32_t>>outputs;};
struct Range {uintptr_t at;size_t bytes;};
inline bool overlaps(Range a,Range b){return a.at<b.at+b.bytes&&b.at<a.at+a.bytes;}
inline bool range(Span s){return s.data&&s.bytes&&uintptr_t(s.data)<=UINTPTR_MAX-s.bytes;}
inline NativeResult<unsigned> validate(const Binding&b){
 if(!live_borrows(b))return NativeError::InvalidBinding;
 if(b.owner.layer>=64||!b.owner.generation||b.readers.empty()||b.readers.size()>9||b.features%32||!b.features)return NativeError::InvalidBinding;
 auto ext=admit({unsigned(b.prep.n),b.batch,b.source_rows,unsigned(b.prep.ldx),b.prep.flags});if(!std::holds_alternative<Extents>(ext))return NativeError::InvalidBinding;auto e=std::get<Extents>(ext);unsigned n=b.prep.n;
 if(b.xbytes<uint64_t(b.source_rows-1)*b.prep.ldx*4+n*4||b.q.bytes<e.q4||b.rq.bytes<e.q4||b.s.bytes<e.s4||b.rs.bytes<e.s4||b.g.bytes<e.g4||b.rg.bytes<e.g4||b.transformed.bytes<e.f32||b.norm.bytes<e.norm||b.status.bytes<b.batch*(n/32)*4||b.rstatus.bytes<b.batch*(n/32)*4)return NativeError::InvalidBinding;
 if(!(b.prep.flags&Store)||b.prep.ldo!=int(n)||b.prep.out_f32!=b.transformed.data||b.prep.out_norm!=b.norm.data||b.prep.q4!=b.q.data||b.prep.s4!=b.s.data||b.prep.g4!=b.g.data||b.prior.size()!=uint64_t(b.batch)*b.features)return NativeError::InvalidBinding;
 if(!((n==6144&&b.features==5120)||(n==5120&&(b.features==17408||b.features==34816||b.features==16384||b.features==14336||b.features==96||b.features==248320))||(n==17408&&b.features==5120)))return NativeError::InvalidBinding;
 if((b.prep.flags&Norm)&&(!b.prep.norm_w||b.normbytes!=n*4))return NativeError::InvalidBinding;
 if((b.prep.flags&(Sign|SignAfter))&&(!b.prep.signs||b.signbytes!=n*4))return NativeError::InvalidBinding;
 if((b.prep.flags&SiluMul)&&(!b.prep.x2||b.x2bytes<uint64_t(b.source_rows-1)*b.prep.ldx*4+n*4))return NativeError::InvalidBinding;
 if((b.prep.flags&GdnNorm)&&(!b.prep.gnorm_w||b.gnormbytes!=512||!b.prep.gz||b.prep.gz_ld<6144||b.gzbytes<uint64_t(b.source_rows-1)*b.prep.gz_ld*4+6144*4))return NativeError::InvalidBinding;
 std::vector<Span>input{{b.prep.x,b.xbytes}};if(b.prep.x2)input.push_back({b.prep.x2,b.x2bytes});if(b.prep.norm_w)input.push_back({b.prep.norm_w,b.normbytes});if(b.prep.signs)input.push_back({b.prep.signs,b.signbytes});if(b.prep.gz)input.push_back({b.prep.gz,b.gzbytes});if(b.prep.gnorm_w)input.push_back({b.prep.gnorm_w,b.gnormbytes});if(b.prep.rowmap)input.push_back({b.prep.rowmap,size_t(b.batch)*4});
 std::vector<Writable>output{b.q,b.s,b.g,b.rq,b.rs,b.rg,b.transformed,b.norm,b.status,b.rstatus};
 for(unsigned i=0;i<b.readers.size();i++){auto&r=b.readers[i];if(r.wbytes<uint64_t(n/128)*(b.features/32)*512||r.sbytes<uint64_t(n/128)*b.features*2||r.outbytes<uint64_t(b.batch)*b.features*4||r.ownerbytes<uint64_t(b.batch)*b.features*4)return NativeError::InvalidBinding;for(unsigned j=0;j<i;j++)if(r.id==b.readers[j].id)return NativeError::InvalidBinding;input.push_back({r.w,r.wbytes});input.push_back({r.s,r.sbytes});output.push_back({r.out,r.outbytes});output.push_back({r.owners0,r.ownerbytes});output.push_back({r.owners1,r.ownerbytes});}
 for(auto i:input)if(!range(i))return NativeError::InvalidBinding;
 for(unsigned i=0;i<output.size();i++){auto a=output[i];if(!range({a.data,a.bytes})||(uintptr_t(a.data)&3))return NativeError::InvalidBinding;for(auto p:input)if(overlaps({uintptr_t(a.data),a.bytes},{uintptr_t(p.data),p.bytes}))return NativeError::InvalidBinding;for(unsigned j=0;j<i;j++)if(overlaps({uintptr_t(a.data),a.bytes},{uintptr_t(output[j].data),output[j].bytes}))return NativeError::InvalidBinding;}
 return n;
}
class Session {
 std::optional<Binding>binding_;std::optional<Observation>words_;bool drained_=false,observed_=false;
 template<class T>bool copy(std::vector<T>&out,const void*data,size_t count){out.resize(count);return binding15::copy(out.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost)==cudaSuccess;}
public:
 Session()=default;Session(const Session&)=delete;Session&operator=(const Session&)=delete;
 bool owns_pending()const{return binding_.has_value();}
 NativeResult<Observation*> run(const Binding&b) try {
  if(binding_)return NativeError::Busy;auto admitted=validate(b);if(auto e=std::get_if<NativeError>(&admitted))return *e;binding_=b;drained_=false;observed_=false;unsigned n=std::get<unsigned>(admitted);
  if(b.prep.rowmap){std::vector<int>map;if(!copy(map,b.prep.rowmap,b.batch))return NativeError::Copy;for(int r:map)if(r<0||unsigned(r)>=b.source_rows)return NativeError::InvalidBinding;}
  if(binding15::fill(b.status.data,0xff,b.status.bytes)!=cudaSuccess||binding15::fill(b.rstatus.data,0xff,b.rstatus.bytes)!=cudaSuccess)return NativeError::Copy;
  if(!live_borrows(b)||!binding15::ok(binding15::active->start(b.observation_id))||!binding15::before(binding15::Effect::launch))return NativeError::Launch;
  k_owned_prep<<<b.batch,n/32>>>(b.prep,(unsigned*)b.status.data);if(binding15::finish(cudaGetLastError())!=cudaSuccess||!binding15::ok(binding15::active->launched(b.observation_id)))return NativeError::Launch;
  h5::PrepArgs residual{};residual.x=(float*)b.transformed.data;residual.ldx=n;residual.n=n;residual.flags=Quant|Residual;residual.q4=(uint8_t*)b.rq.data;residual.s4=(uint8_t*)b.rs.data;residual.g4=(float*)b.rg.data;residual.ref_q4=(uint8_t*)b.q.data;residual.ref_s4=(uint8_t*)b.s.data;residual.ref_g4=(float*)b.g.data;
  if(!live_borrows(b)||!binding15::ok(binding15::active->start(b.observation_id+1))||!binding15::before(binding15::Effect::launch))return NativeError::Launch;
  k_owned_prep<<<b.batch,n/32>>>(residual,(unsigned*)b.rstatus.data);if(binding15::finish(cudaGetLastError())!=cudaSuccess||!binding15::ok(binding15::active->launched(b.observation_id+1)))return NativeError::Launch;
  for(auto&r:b.readers){if(binding15::copy(r.out,b.prior.data(),b.prior.size_bytes(),cudaMemcpyHostToDevice)!=cudaSuccess)return NativeError::Copy;
   if(binding15::fill(r.owners0,0xff,r.ownerbytes)!=cudaSuccess||binding15::fill(r.owners1,0xff,r.ownerbytes)!=cudaSuccess)return NativeError::Copy;
   auto launch=[&](const uint8_t*q,const uint8_t*s,const float*g,unsigned*owners,uint64_t id){constexpr unsigned SM=WsGeom<4,1,32,4>::SMEM;if(!live_borrows(b)||!binding15::ok(binding15::active->start(id))||binding15::attribute(k_owned_gemm_fp4<4,1,32,1,4>,cudaFuncAttributeMaxDynamicSharedMemorySize,SM)!=cudaSuccess||!binding15::before(binding15::Effect::launch))return false;k_owned_gemm_fp4<4,1,32,1,4><<<dim3((b.batch+31)/32,(b.features/32+3)/4,1),160,SM>>>(q,s,g,b.batch,n,r.w,(uint8_t*)r.s,b.features,r.out,b.features,nullptr,owners);return binding15::finish(cudaGetLastError())==cudaSuccess&&binding15::ok(binding15::active->launched(id));};
   uint64_t id=b.observation_id+2+2*(r.id-1);
   if(!launch((uint8_t*)b.q.data,(uint8_t*)b.s.data,(float*)b.g.data,r.owners0,id)||!launch((uint8_t*)b.rq.data,(uint8_t*)b.rs.data,(float*)b.rg.data,r.owners1,id+1))return NativeError::Launch;
  }
  if(binding15::fence()!=cudaSuccess)return NativeError::Fence;drained_=true;words_.emplace();auto&o=*words_;o.owner=b.owner;auto e=std::get<Extents>(admit({n,b.batch,b.source_rows,unsigned(b.prep.ldx),b.prep.flags}));
  binding15::active->select(b.observation_id);
  if(!copy(o.q,b.q.data,e.q4)||!copy(o.s,b.s.data,e.s4)||!copy(o.g,b.g.data,b.batch)||!copy(o.transformed,b.transformed.data,b.batch*n)||!copy(o.norm,b.norm.data,b.batch*n)||!copy(o.status,b.status.data,b.batch*(n/32)))return NativeError::Observe;
  binding15::active->select(b.observation_id+1);
  if(!copy(o.rq,b.rq.data,e.q4)||!copy(o.rs,b.rs.data,e.s4)||!copy(o.rg,b.rg.data,b.batch)||!copy(o.rstatus,b.rstatus.data,b.batch*(n/32)))return NativeError::Observe;
  for(unsigned s:o.status)if(s)return NativeError::Status;for(unsigned s:o.rstatus)if(s)return NativeError::Status;
  for(auto&r:b.readers){uint64_t id=b.observation_id+2+2*(r.id-1);binding15::active->select(id);o.outputs.emplace_back();if(!copy(o.outputs.back(),r.out,uint64_t(b.batch)*b.features))return NativeError::Observe;for(auto*status:{r.owners0,r.owners1}){binding15::active->select(id++);o.reader_status.emplace_back();if(!copy(o.reader_status.back(),status,uint64_t(b.batch)*b.features))return NativeError::Observe;for(unsigned s:o.reader_status.back())if(s)return NativeError::Status;}}observed_=true;return &o;
 } catch(const std::bad_alloc&){return NativeError::Allocation;}
 NativeResult<unsigned> release(){if(!binding_||!words_||!drained_||!observed_)return NativeError::Release;words_.reset();binding_.reset();observed_=false;return 0u;}
 NativeResult<unsigned> abort(){if(!binding_)return 0u;if(binding15::cleanup_fence()!=bonsai14::Outcome::success)return NativeError::Fence;drained_=true;words_.reset();binding_.reset();observed_=false;return 0u;}
};
}
