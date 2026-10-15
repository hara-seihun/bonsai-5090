#pragma once
#include "binding.hpp"
#include <cuda_runtime.h>
#include <array>
namespace binding15 {
inline Binding* active=nullptr;
enum class Evidence {live,released,unknown};
struct EvidenceRecord {uint64_t handle;Evidence state;bonsai14::Owner owner;unsigned stream;std::array<unsigned,5>domain;};
inline unsigned current_stream=0;inline std::array<unsigned,5>current_domain={0,1,1,1,0};
inline std::vector<EvidenceRecord> evidence;
inline size_t evidence_count=0;
inline EvidenceRecord* evidence_record(uint64_t h){for(size_t i=evidence_count;i;i--)if(evidence[i-1].handle==h)return &evidence[i-1];return nullptr;}
inline std::optional<Evidence> known(uint64_t h){auto*p=evidence_record(h);if(!p)return std::nullopt;return p->state;}
inline uint64_t identity(const void*p){return uint64_t(reinterpret_cast<uintptr_t>(p));}
inline bool before(Effect effect,const void*p=nullptr){return active&&active->record_before(effect,p?std::optional<uint64_t>{identity(p)}:std::nullopt);}
inline cudaError_t finish(cudaError_t result){if(!active)return cudaErrorInvalidValue;auto outcome=active->finish(result==cudaSuccess?Outcome::success:Outcome::uncertain);return outcome==Outcome::success?cudaSuccess:(result==cudaSuccess?cudaErrorUnknown:result);}
inline cudaError_t allocate(void**out,size_t n){if(!out||!n||!active)return cudaErrorInvalidValue;*out=nullptr;if(evidence_count==evidence.capacity())return cudaErrorMemoryAllocation;cudaError_t result=cudaSuccess;
 auto s=active->acquire([&](){evidence.push_back({0,Evidence::unknown,active->events().back().owner,current_stream,current_domain});auto&record=evidence[evidence_count++];active->details(std::nullopt,n,std::nullopt);result=::cudaMalloc(out,n);record.handle=identity(*out);record.state=result==cudaSuccess?Evidence::live:Evidence::unknown;return NativeAcquisition{result==cudaSuccess?Outcome::success:Outcome::uncertain,result==cudaSuccess?Acquisition{Acquired{identity(*out)}}:Acquisition{Uncertain{*out?std::optional<uint64_t>{identity(*out)}:std::nullopt}}};});
 return ok(s)?cudaSuccess:(result==cudaSuccess?cudaErrorUnknown:result);
}
template<class T>inline cudaError_t allocate(T**out,size_t n){return allocate(reinterpret_cast<void**>(out),n);}
template<class T>inline Result<ValidationHandle> allocate_validation(T**out,size_t n,Owner owner){
 if(!out||!n||!active)return Error::handle;*out=nullptr;if(evidence_count==evidence.capacity())return Error::capacity;
 return active->acquire_validation(owner,ByteRange{0,n,n},[&](){evidence.push_back({0,Evidence::unknown,owner,current_stream,current_domain});auto&record=evidence[evidence_count++];auto e=::cudaMalloc(out,n);record.handle=identity(*out);record.state=e==cudaSuccess?Evidence::live:Evidence::unknown;return NativeAcquisition{e==cudaSuccess?Outcome::success:Outcome::uncertain,e==cudaSuccess?Acquisition{Acquired{identity(*out)}}:Acquisition{Uncertain{*out?std::optional<uint64_t>{identity(*out)}:std::nullopt}}};});
}
inline cudaError_t copy(void*d,const void*s,size_t n,cudaMemcpyKind kind){Effect effect=kind==cudaMemcpyHostToDevice?Effect::upload:kind==cudaMemcpyDeviceToHost?Effect::observe:Effect::copy;auto result=active->effect(effect,identity(kind==cudaMemcpyDeviceToHost?s:d),[&](){active->details(identity(s),n,unsigned(kind));return ::cudaMemcpy(d,s,n,kind)==cudaSuccess?Outcome::success:Outcome::uncertain;});return result==Outcome::success?cudaSuccess:cudaErrorUnknown;}
inline cudaError_t fill(void*d,int value,size_t n){auto result=active->effect(Effect::fill,identity(d),[&](){active->details(std::nullopt,n,unsigned(value));return ::cudaMemset(d,value,n)==cudaSuccess?Outcome::success:Outcome::uncertain;});return result==Outcome::success?cudaSuccess:cudaErrorUnknown;}
template<class Kernel,class... Args>inline cudaError_t launch_ex(const cudaLaunchConfig_t*cfg,Kernel kernel,Args&&...args){if(!before(Effect::launch))return cudaErrorUnknown;return finish(::cudaLaunchKernelEx(cfg,kernel,std::forward<Args>(args)...));}
inline cudaError_t fence(){if(!before(Effect::fence))return cudaErrorUnknown;return finish(::cudaDeviceSynchronize());}
template<class Kernel>inline cudaError_t attribute(Kernel kernel,cudaFuncAttribute attr,int value){if(!before(Effect::attribute))return cudaErrorUnknown;active->details(std::nullopt,size_t(value),unsigned(attr));return finish(::cudaFuncSetAttribute(kernel,attr,value));}
// Cleanup can proceed while forward execution is faulted. Exact releases are reconciled in Protocol.
inline Outcome release(HandleKind kind,uint64_t native){if(kind!=HandleKind::allocation||!native)return Outcome::failure;auto*record=evidence_record(native);if(!record||record->stream!=current_stream)return Outcome::failure;record->state=Evidence::unknown;auto e=::cudaFree(reinterpret_cast<void*>(uintptr_t(native)));if(e==cudaSuccess)record->state=Evidence::released;return e==cudaSuccess?Outcome::success:Outcome::uncertain;}
inline Outcome cleanup_fence(){return ::cudaDeviceSynchronize()==cudaSuccess?Outcome::success:Outcome::uncertain;}
}
