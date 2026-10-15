#pragma once
#include "format.hpp"
#include <map>
#include <optional>
namespace halo6000 {
struct MappedFile {
 void*base=nullptr;size_t size=0;SourceIdentity identity;
 ~MappedFile();
 void discard(const void*data,size_t bytes)const;
 MappedFile()=default;MappedFile(const MappedFile&)=delete;MappedFile&operator=(const MappedFile&)=delete;
 static Result<std::shared_ptr<MappedFile>>open(const std::string&path);
};
struct GgufValue {
 uint32_t type=0,element_type=0;uint64_t count=0;
 std::vector<uint8_t>raw;std::vector<std::string>strings;
 std::optional<int64_t>integer()const;
 std::optional<double>number()const;
};
struct GgufTensor {std::string name;std::vector<uint64_t>shape;uint32_t type=0;uint64_t offset=0;size_t bytes=0;const uint8_t*data=nullptr;uint64_t elements()const{uint64_t n=1;for(auto d:shape)n*=d;return n;}};
struct Gguf {
 std::shared_ptr<MappedFile>storage;std::map<std::string,GgufValue>metadata;std::vector<GgufTensor>tensors;
 static Result<Gguf>open(const std::string&path);
 const GgufTensor*find(const std::string&name)const;
 Status validate_model()const;
};
Result<size_t>tensor_bytes(uint32_t type,const std::vector<uint64_t>&shape);
} // namespace halo6000
