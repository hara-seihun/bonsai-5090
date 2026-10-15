#pragma once
#include "dict8_codec.hpp"
#include "dict8_binding.hpp"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <variant>
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib,"bcrypt.lib")
#else
#include <openssl/evp.h>
#endif
namespace h5::dict8::cache {
using Digest=std::array<uint8_t,32>;
enum class Error : int { Unset=1, Missing, Io, Header, Version, Shape, ModelIdentity, Semantics, Truncated, Corrupt, LayerOrder, Crypto, Budget, Exists, Publication, Allocation, ModelLayout, State, Staged, PublishedDurabilityUnknown, PublishedCleanupRequired, StagedCleanupRequired, Threads, Initialization, Protection, StaleIdentity, SubstitutedIdentity, UnsupportedStorage, ModelInUse, Mapping, AdmissionQuery, RemoteStorage, ReparseStorage, ModelDirectory, ModelDeletePending };
inline const char* name(Error e) {
 switch(e) {
#define D8E(x) case Error::x:return #x;
 D8E(Unset) D8E(Missing) D8E(Io) D8E(Header) D8E(Version) D8E(Shape) D8E(ModelIdentity) D8E(Semantics) D8E(Truncated) D8E(Corrupt) D8E(LayerOrder) D8E(Crypto) D8E(Budget) D8E(Exists) D8E(Publication) D8E(Allocation) D8E(ModelLayout) D8E(State) D8E(Staged) D8E(PublishedDurabilityUnknown) D8E(PublishedCleanupRequired) D8E(StagedCleanupRequired) D8E(Threads) D8E(Initialization) D8E(Protection) D8E(StaleIdentity) D8E(SubstitutedIdentity) D8E(UnsupportedStorage) D8E(ModelInUse) D8E(Mapping) D8E(AdmissionQuery) D8E(RemoteStorage) D8E(ReparseStorage) D8E(ModelDirectory) D8E(ModelDeletePending)
#undef D8E
 } return "invalid-error-enum";
}
template<class T> using Result=std::variant<T,Error>;
inline uint64_t tick(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
inline Result<Digest> hash(const void* data,size_t n) {
 Digest out{};
#ifdef _WIN32
 BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE h=nullptr;
 if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)return Error::Crypto;
 auto status=BCryptCreateHash(alg,&h,nullptr,0,nullptr,0,0);
 if(status>=0)for(size_t off=0;off<n;) {auto size=ULONG((n-off)>0x1000000?0x1000000:n-off);status=BCryptHashData(h,(PUCHAR)data+off,size,0);if(status<0)break;off+=size;}
 if(status>=0)status=BCryptFinishHash(h,out.data(),32,0);
 if(h)BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(alg,0);
 if(status<0)return Error::Crypto;
#else
 auto* h=EVP_MD_CTX_new();if(!h)return Error::Crypto;unsigned count=0;
 bool ok=EVP_DigestInit_ex(h,EVP_sha256(),nullptr)==1&&EVP_DigestUpdate(h,data,n)==1&&EVP_DigestFinal_ex(h,out.data(),&count)==1;
 EVP_MD_CTX_free(h);if(!ok||count!=32)return Error::Crypto;
#endif
 return out;
}
inline std::string hex(const Digest& d){const char* c="0123456789abcdef";std::string s;for(auto x:d){s+=c[x>>4];s+=c[x&15];}return s;}
inline Digest unhex(const char* s){Digest d{};for(size_t i=0;i<32;i++){auto v=[](char c){return c<='9'?c-'0':c-'a'+10;};d[i]=uint8_t(16*v(s[2*i])+v(s[2*i+1]));}return d;}
constexpr uint32_t layers=64,K=5120,N=34816;
constexpr uint64_t layer_bytes=uint64_t(N/32)*(K/128)*832,header_bytes=128+layers*32,file_bytes=header_bytes+layers*layer_bytes;
using Header=std::array<uint8_t,header_bytes>;
inline void put32(uint8_t* p,uint32_t v){for(int i=0;i<4;i++)p[i]=uint8_t(v>>(8*i));}
inline uint32_t get32(const uint8_t* p){uint32_t v=0;for(int i=0;i<4;i++)v|=uint32_t(p[i])<<(8*i);return v;}
inline Header header(){Header h{};std::memcpy(h.data(),"D8CACHE1",8);put32(h.data()+8,1);put32(h.data()+12,layers);put32(h.data()+16,K);put32(h.data()+20,N);put32(h.data()+24,832);put32(h.data()+28,1664);auto m=unhex(model_sha256),s=unhex(semantics_sha256);std::memcpy(h.data()+32,m.data(),32);std::memcpy(h.data()+64,s.data(),32);std::memcpy(h.data()+128,payload_table,layers*32);return h;}
inline Result<bool> validate_header(const Header& h){auto expected=header();if(std::memcmp(h.data(),expected.data(),8))return Error::Header;if(get32(h.data()+8)!=1)return Error::Version;if(std::memcmp(h.data()+12,expected.data()+12,20))return Error::Shape;if(std::memcmp(h.data()+32,expected.data()+32,32))return Error::ModelIdentity;if(std::memcmp(h.data()+64,expected.data()+64,32))return Error::Semantics;for(int i=96;i<128;i++)if(h[i])return Error::Header;auto table=hash(h.data()+128,layers*32);if(auto* e=std::get_if<Error>(&table))return *e;if(std::get<Digest>(table)!=unhex(payload_table_sha256))return Error::Corrupt;return true;}
struct CloseFile {void operator()(FILE* f)const{if(f)std::fclose(f);}};
using File=std::unique_ptr<FILE,CloseFile>;
struct Costs {uint64_t model_bytes=0,model_hash_ns=0,open_header_ns=0,read_bytes=0,read_ns=0,hash_ns=0,checked_layers=0,allocation_ns=0,finish_ns=0;};
struct View {const uint8_t* ptr;size_t size_;const uint8_t* data()const{return ptr;}size_t size()const{return size_;}};
}
