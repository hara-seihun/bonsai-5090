#if defined(NDEBUG)
#error Dict8 CPU assertions must remain enabled
#endif
#include "dict8_cache.hpp"
#include <cassert>
#include <cerrno>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <openssl/evp.h>
namespace c=h5::dict8::cache;
enum class Fault { None, Allocation, Protection, ReadIo, ReadTruncated, ExtraByte, Crypto };
static Fault fault=Fault::None;static int live_mappings=0;static uint64_t reads=0,hashes=0;
extern "C" void* __real_mmap(void*,size_t,int,int,int,off_t);
extern "C" void* __wrap_mmap(void* p,size_t n,int prot,int flags,int fd,off_t offset){if(fault==Fault::Allocation){errno=ENOMEM;return MAP_FAILED;}auto out=__real_mmap(p,n,prot,flags,fd,offset);if(out!=MAP_FAILED)++live_mappings;return out;}
extern "C" int __real_munmap(void*,size_t);
extern "C" int __wrap_munmap(void* p,size_t n){int out=__real_munmap(p,n);if(!out)--live_mappings;return out;}
extern "C" int __real_mprotect(void*,size_t,int);
extern "C" int __wrap_mprotect(void* p,size_t n,int prot){if(fault==Fault::Protection){errno=EACCES;return -1;}return __real_mprotect(p,n,prot);}
extern "C" ssize_t __real_read(int,void*,size_t);
extern "C" ssize_t __wrap_read(int fd,void* p,size_t n){++reads;if(fault==Fault::ReadIo){errno=EIO;return -1;}if(fault==Fault::ReadTruncated)return 0;if(fault==Fault::ExtraByte&&n==1){*static_cast<char*>(p)=0;return 1;}return __real_read(fd,p,n);}
extern "C" int __real_EVP_DigestUpdate(EVP_MD_CTX*,const void*,size_t);
extern "C" int __wrap_EVP_DigestUpdate(EVP_MD_CTX* h,const void* p,size_t n){++hashes;if(fault==Fault::Crypto)return 0;return __real_EVP_DigestUpdate(h,p,n);}
int main(int argc,char** argv){assert(argc==2);
 unsigned tested=0;
 for(auto test: {std::pair{Fault::Allocation,c::Error::Allocation},std::pair{Fault::Protection,c::Error::Protection},std::pair{Fault::ReadIo,c::Error::Io},std::pair{Fault::ReadTruncated,c::Error::ModelIdentity},std::pair{Fault::ExtraByte,c::Error::ModelIdentity},std::pair{Fault::Crypto,c::Error::Crypto}}){
  fault=test.first;auto before=hashes;auto r=c::open_model(argv[1]);assert(std::holds_alternative<c::Error>(r)&&std::get<c::Error>(r)==test.second);assert(live_mappings==0);if(fault!=Fault::Crypto)assert(hashes==before);++tested;
 }
 fault=Fault::None;auto r=c::open_model(argv[1]);assert(std::holds_alternative<std::unique_ptr<c::ExactModel>>(r)&&live_mappings==1);std::get<std::unique_ptr<c::ExactModel>>(r).reset();assert(live_mappings==0);
 std::cout<<"{\"injected_linux_boundary_failures\":"<<tested<<",\"failed_mapping_cleanup\":true,\"failed_acquisition_never_hashes\":true,\"recovered_after_faults\":true,\"gpu_work\":false}\n";
}
