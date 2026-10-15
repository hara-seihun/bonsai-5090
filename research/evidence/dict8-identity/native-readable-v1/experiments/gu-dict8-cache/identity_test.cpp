#include "dict8_binding.hpp"
#include <cstdint>
#include <cstddef>
static uint64_t observed_identity_scans=0;
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
static NTSTATUS counted_hash(BCRYPT_HASH_HANDLE h,PUCHAR p,ULONG n,ULONG flags){
 if(n==h5::dict8::cache::model_size)++observed_identity_scans;
 return BCryptHashData(h,p,n,flags);
}
#define BCryptHashData counted_hash
#else
#include <openssl/evp.h>
extern "C" int __real_EVP_DigestUpdate(EVP_MD_CTX*,const void*,size_t);
extern "C" int __wrap_EVP_DigestUpdate(EVP_MD_CTX* h,const void* p,size_t n){
 if(n==h5::dict8::cache::model_size)++observed_identity_scans;
 return __real_EVP_DigestUpdate(h,p,n);
}
#endif
#include "dict8_cache.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <type_traits>
#ifndef _WIN32
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#endif
namespace c=h5::dict8::cache;
using namespace h5;
template<class T> T take(c::Result<T> r){assert(std::holds_alternative<T>(r));return std::move(std::get<T>(r));}
template<class T> void error(c::Result<T> r,c::Error e){assert(std::holds_alternative<c::Error>(r));assert(std::get<c::Error>(r)==e);}
struct Failure {c::Error error;};
template<class T> T cache_take(c::Result<T> r){if(auto* e=std::get_if<c::Error>(&r))throw Failure{*e};return std::move(std::get<T>(r));}
static uint64_t device_boundary_calls=0;
struct EngineHarness {
 Gguf g;std::unique_ptr<c::ExactModel> dict8_model;std::unique_ptr<c::Reader> dict8_cache;uint64_t dict8_preparse_ns=0;
 @SELECTOR@
 EngineHarness(const char* path,bool dict8_enabled){
 @BOOTSTRAP@
 ++device_boundary_calls;
 }
};
static void env(const char* p){if(p)setenv("H5_GU_DICT8_CACHE",p,1);else unsetenv("H5_GU_DICT8_CACHE");}
static void write(const std::string& path,const std::vector<uint8_t>& v){std::ofstream f(path,std::ios::binary|std::ios::trunc);assert(f);f.write(reinterpret_cast<const char*>(v.data()),v.size());assert(f);}
int main(int argc,char** argv){
 assert(argc==3);std::string root=argv[1],good=root+"/model.gguf",cache=root+"/weights.d8",bad=root+"/bad.gguf",fault=root+"/fault.d8";
 static_assert(!std::is_default_constructible_v<c::ExactModel::Identity>);
 static_assert(!std::is_constructible_v<c::ExactModel::Identity,const void*,size_t>);
 static_assert(!std::is_copy_constructible_v<c::ExactModel>);
 static_assert(!std::is_move_constructible_v<c::ExactModel>);
 error(c::open_model(nullptr),c::Error::Unset);error(c::open_model(""),c::Error::Unset);error(c::open_model((root+"/absent").c_str()),c::Error::Missing);
 write(bad,{0,1,2});error(c::open_model(bad.c_str()),c::Error::ModelIdentity);
 auto bytes=[](const std::string& path){std::ifstream f(path,std::ios::binary);return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),{});};auto original=bytes(good);auto wrong=original;wrong[0]^=255;write(bad,wrong);
 auto before=observed_identity_scans;error(c::open_model(bad.c_str()),c::Error::ModelIdentity);assert(observed_identity_scans==before+1); // invalid magic never reaches the unsafe parser
 c::StorageFacts facts{1,87,L"NTFS",L"\\\\?\\Volume{fixture}\\model.gguf",3,0x20,0,false,false,int64_t(c::model_size)};assert(take(c::admit_storage(facts)));
 unsigned admission_faults=0;
 auto reject=[&](c::StorageFacts f,c::Error e){error(c::admit_storage(f),e);++admission_faults;};
 auto f=facts;f.file_type=2;reject(f,c::Error::UnsupportedStorage);
 f=facts;f.remote_query_error=0;reject(f,c::Error::RemoteStorage);
 for(auto code:{1U,5U,50U,122U,123U}){f=facts;f.remote_query_error=code;reject(f,c::Error::AdmissionQuery);}
 f=facts;f.filesystem=L"ReFS";reject(f,c::Error::UnsupportedStorage);f.filesystem=L"ntfs";reject(f,c::Error::UnsupportedStorage);
 f=facts;f.volume_path=L"\\\\server\\share\\model";reject(f,c::Error::UnsupportedStorage);f.volume_path=L"\\\\?\\Volume{broken";reject(f,c::Error::UnsupportedStorage);
 for(auto type:{0U,1U,2U,4U,5U,6U}){f=facts;f.drive_type=type;reject(f,c::Error::UnsupportedStorage);}
 for(auto attr:{0x400U,0x1000U,0x40000U,0x400000U}){f=facts;f.attributes|=attr;reject(f,c::Error::ReparseStorage);}
 f=facts;f.reparse_tag=1;reject(f,c::Error::ReparseStorage);f=facts;f.directory=true;reject(f,c::Error::ModelDirectory);f=facts;f.delete_pending=true;reject(f,c::Error::ModelDeletePending);
 for(auto size:{int64_t(-1),int64_t(0),int64_t(c::model_size-1),int64_t(c::model_size+1)}){f=facts;f.bytes=size;reject(f,c::Error::ModelIdentity);}
#ifndef _WIN32
 int fd=::open(good.c_str(),O_RDWR);assert(fd>=0);auto writable=mmap(nullptr,original.size(),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);assert(writable!=MAP_FAILED);
#endif
 env(cache.c_str());before=observed_identity_scans;
 auto engine=std::make_unique<EngineHarness>(good.c_str(),true);
 assert(observed_identity_scans==before+1&&engine->dict8_model->costs().identity_scans==1);
 assert(engine->g.base==nullptr&&&engine->model()==&engine->dict8_model->gguf());
 const auto& mc=engine->dict8_model->costs();assert(mc.identity_bytes==original.size());
#ifdef _WIN32
 assert(mc.storage==uint64_t(c::ModelStorage::WindowsRetainedLocalNtfs)&&mc.mapped_bytes==original.size()&&mc.copied_bytes==0&&mc.allocated_bytes==0&&mc.source_read_bytes==0);
#else
 assert(mc.storage==uint64_t(c::ModelStorage::LinuxPrivateSnapshot)&&mc.copied_bytes==original.size()&&mc.source_read_bytes==original.size()&&mc.allocated_bytes>=original.size()&&mc.mapped_bytes==0);
#endif
 auto id=engine->dict8_model->identity();auto other=take(c::open_model(good.c_str()));
 error(c::Reader::open(cache.c_str(),*other,id),c::Error::SubstitutedIdentity);
 auto stale=other->identity();other.reset();error(c::Reader::open(cache.c_str(),*engine->dict8_model,stale),c::Error::StaleIdentity);
 error(c::Reader::open(nullptr,*engine->dict8_model,id),c::Error::Unset);error(c::Reader::open("",*engine->dict8_model,id),c::Error::Unset);
 error(c::Reader::open((root+"/absent-cache").c_str(),*engine->dict8_model,id),c::Error::Missing);
 auto good_cache=bytes(cache);auto mut=good_cache;
 for(auto entry:std::vector<std::pair<size_t,c::Error>>{{0,c::Error::Header},{8,c::Error::Version},{12,c::Error::Shape},{32,c::Error::ModelIdentity},{64,c::Error::Semantics},{96,c::Error::Header},{128,c::Error::Corrupt},{c::header_bytes,c::Error::Corrupt}}){mut=good_cache;mut[entry.first]^=1;write(fault,mut);error(c::Reader::open(fault.c_str(),*engine->dict8_model,id),entry.second);}
 mut=good_cache;for(int j=0;j<32;j++)std::swap(mut[128+j],mut[160+j]);write(fault,mut);error(c::Reader::open(fault.c_str(),*engine->dict8_model,id),c::Error::Corrupt);
 mut=good_cache;mut.pop_back();write(fault,mut);error(c::Reader::open(fault.c_str(),*engine->dict8_model,id),c::Error::Truncated);
 mut=good_cache;mut.push_back(0);write(fault,mut);error(c::Reader::open(fault.c_str(),*engine->dict8_model,id),c::Error::Truncated);
 assert(engine->dict8_cache->costs.model_hash_ns==0&&engine->dict8_cache->costs.model_bytes==original.size());
 before=observed_identity_scans;
 auto another=take(c::Reader::open(cache.c_str(),*engine->dict8_model,id));assert(observed_identity_scans==before);
 error(another->layer(1),c::Error::LayerOrder);error(another->layer(0),c::Error::State);another.reset();
 another=take(c::Reader::open(cache.c_str(),*engine->dict8_model,id));error(another->finish(),c::Error::LayerOrder);error(another->finish(),c::Error::State);another.reset();
 const uint8_t* tensor=engine->model().t("weight").data;assert(*tensor==42);
#ifdef _WIN32
 auto writer=CreateFileA(good.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);assert(writer==INVALID_HANDLE_VALUE&&GetLastError()==ERROR_SHARING_VIOLATION);
 assert(!DeleteFileA(good.c_str())&&GetLastError()==ERROR_SHARING_VIOLATION);
#else
 // MAP_SHARED writable alias predates the admission and remains writable: snapshot cannot alias it.
 static_cast<uint8_t*>(writable)[original.size()-1]^=1;assert(*tensor==42);assert(tensor[3]==0);
 munmap(writable,original.size());close(fd);
 wrong=original;wrong[0]=0;write(good,wrong);assert(*tensor==42);
 auto pid=fork();assert(pid>=0);if(!pid){struct rlimit limit{0,0};setrlimit(RLIMIT_CORE,&limit);*const_cast<uint8_t*>(tensor)=0;_exit(0);}int status;waitpid(pid,&status,0);assert(WIFSIGNALED(status)&&WTERMSIG(status)==SIGSEGV);
#endif
 auto held_reader=std::move(engine->dict8_cache);engine.reset();assert(*tensor==42);
#ifdef _WIN32
 auto still_denied=CreateFileA(good.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);assert(still_denied==INVALID_HANDLE_VALUE&&GetLastError()==ERROR_SHARING_VIOLATION);
#endif
 auto reads_before=observed_identity_scans;for(uint32_t i=0;i<c::layers;i++)assert(take(held_reader->layer(i)).size()==c::layer_bytes);assert(take(held_reader->finish()));error(held_reader->layer(0),c::Error::State);error(held_reader->finish(),c::Error::State);assert(observed_identity_scans==reads_before);
 held_reader.reset(); // now no snapshot/deny-write handle survives
#ifdef _WIN32
 auto released=CreateFileA(good.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);assert(released!=INVALID_HANDLE_VALUE);CloseHandle(released);
#endif
 write(good,original);
 auto replacement=take(c::open_model(good.c_str()));error(c::Reader::open(cache.c_str(),*replacement,id),c::Error::StaleIdentity);replacement.reset();
 // Closed writer handle but retained writable section/view is an admission failure on local NTFS.
#ifdef _WIN32
 HANDLE writer_file=CreateFileA(good.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);assert(writer_file!=INVALID_HANDLE_VALUE);
 HANDLE section=CreateFileMappingA(writer_file,nullptr,PAGE_READWRITE,0,0,nullptr);assert(section);auto view=MapViewOfFile(section,FILE_MAP_WRITE,0,0,0);assert(view);CloseHandle(writer_file);
 error(c::open_model(good.c_str()),c::Error::ModelInUse);UnmapViewOfFile(view);error(c::open_model(good.c_str()),c::Error::ModelInUse);CloseHandle(section);
 auto recovered=take(c::open_model(good.c_str()));recovered.reset();
#endif
 env(nullptr);auto calls=device_boundary_calls;
 try{EngineHarness failed(good.c_str(),true);assert(false);}catch(const Failure& e){assert(e.error==c::Error::Unset);}assert(device_boundary_calls==calls);
 env(fault.c_str());try{EngineHarness failed(good.c_str(),true);assert(false);}catch(const Failure& e){assert(e.error==c::Error::Truncated);}assert(device_boundary_calls==calls);
 before=observed_identity_scans;EngineHarness unset(good.c_str(),false);assert(observed_identity_scans==before&&unset.g.base!=nullptr&&&unset.model()==&unset.g&&device_boundary_calls==calls+1);
 std::cout<<"{\"gpu_work\":false,\"injected_admission_faults\":"<<admission_faults<<",\"fixture_model_bytes\":"<<original.size()<<",\"actual_bootstrap_model_hash_scans\":1,\"reader_additional_model_hash_scans\":0,\"source_mutation_or_write_denial\":true,\"retained_reader_lifetime\":true,\"stale_and_substituted_capabilities\":true,\"unforgeable_api_static_asserts\":true,\"hash_before_unsafe_parse\":true,\"cache_failure_before_device_boundary\":true,\"payload_and_header_faults\":true,\"terminal_order_failures\":true,\"disabled_original_path\":true,\"native_cached_startup_ns\":null}\n";
}
