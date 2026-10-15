#pragma once
#include "model.hpp"
#include <string>
#include <variant>
#include <vector>
#include <memory>
namespace halo6000 {
enum class ErrorCode {Io,Truncated,Invalid,Unsupported,Geometry,CacheMismatch,Cuda,Capacity};
struct Error {ErrorCode code;std::string message;};
template<class T>class Result {
 std::variant<T,Error> state_;
public:
 Result(T v):state_(std::move(v)){} Result(Error e):state_(std::move(e)){}
 bool ok()const{return std::holds_alternative<T>(state_);}
 T&value(){return std::get<T>(state_);}const T&value()const{return std::get<T>(state_);}
 const Error&error()const{return std::get<Error>(state_);}
};
using Status=Result<std::monostate>;
inline Status success(){return std::monostate{};}
inline constexpr uint32_t CacheVersion=1;
inline constexpr uint64_t GeometryTag=0x28300a0802000001ULL;
inline constexpr size_t PaneTileBlocks=16,PaneTileBytes=544,PaneCodesBytes=512;
inline constexpr size_t Q8GroupBytes=136,Q4GroupBytes=72;
inline size_t pane_bytes(size_t blocks){return (blocks+15)/16*PaneTileBytes;}
inline const uint32_t* pane_words(const void*data,size_t block){return reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(data)+(block/16)*544+(block%16)*32);}
inline const uint16_t* pane_scale(const void*data,size_t block){return reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(data)+(block/16)*544+512+(block%16)*2);}
struct SourceIdentity {uint64_t device=0,inode=0,size=0,mtime_ns=0,ctime_ns=0;bool operator==(const SourceIdentity&o)const{return device==o.device&&inode==o.inode&&size==o.size&&mtime_ns==o.mtime_ns&&ctime_ns==o.ctime_ns;}};
struct Provenance {std::string source_path,cache_path;SourceIdentity source;uint32_t version=CacheVersion;uint64_t geometry=GeometryTag;};
struct MappedFile;
struct HostTensor {
 std::string name;std::vector<uint64_t>shape;uint32_t source_type=0;
 WeightFormat format=WeightFormat::F32;size_t bytes=0;const uint8_t*data=nullptr;
 uint64_t elements()const{uint64_t n=1;for(auto d:shape)n*=d;return n;}
 uint64_t rows()const{return elements()/shape[0];}
 MatrixView view()const{
  MatrixView v;v.data=data;v.k=shape[0];v.n=shape.size()>1?shape[1]:1;v.experts=shape.size()>2?shape[2]:1;v.format=format;v.scale_group=format==WeightFormat::F32?0:(format==WeightFormat::Q8Grouped128||format==WeightFormat::Q4_0)?32:format==WeightFormat::Q2K?16:128;v.bytes=bytes;
  v.row_stride=format==WeightFormat::F32?v.k*4:format==WeightFormat::Q8Grouped128?v.k/128*136:format==WeightFormat::Q4_0?v.k/128*Q4GroupBytes:format==WeightFormat::Q2K?v.k/256*84:0;
  v.expert_stride=v.experts>1?(format==WeightFormat::Pane2?uint64_t(v.n)*v.k/128*34:uint64_t(v.n)*v.row_stride):0;
  return v;
 }
};
struct HostImage {
 Provenance provenance;std::shared_ptr<MappedFile>storage;std::vector<HostTensor>tensors;
 const HostTensor*find(const std::string&name)const{for(auto&t:tensors)if(t.name==name)return &t;return nullptr;}
};
uint32_t half_to_float_bits(uint16_t h);
Status repack_ptq(const uint8_t*source,size_t blocks,uint8_t*dest,size_t bytes);
Status repack_q8(const uint8_t*source,size_t groups128,uint8_t*dest,size_t bytes);
Status convert_small(const uint8_t*source,uint32_t type,size_t elements,uint8_t*dest,size_t bytes);
} // namespace halo6000
