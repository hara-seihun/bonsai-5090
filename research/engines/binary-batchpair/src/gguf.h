#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
namespace h5 {
struct GTensor { std::string name; std::vector<uint64_t> ne; uint32_t type; uint64_t off; const uint8_t* data; size_t bytes; };
struct GValue { uint32_t type = 0, etype = 0; std::vector<std::string> strs; std::vector<int64_t> ints; std::vector<double> nums; };
struct Gguf {
  void* base = nullptr; size_t size = 0;
  std::map<std::string, GValue> kv; std::vector<GTensor> tensors; std::map<std::string, size_t> index;
  void open(const std::string& path);
  const GTensor& t(const std::string& n) const;
  bool has(const std::string& n) const { return index.count(n); }
  ~Gguf();
};
}
