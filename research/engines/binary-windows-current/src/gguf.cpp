#include "platform.h"
#include "gguf.h"
#include <stdexcept>
#include <cstring>
namespace h5 {
static size_t tbytes(uint32_t t, uint64_t n) {
  switch (t) { case 0: return n * 4; case 1: case 30: return n * 2; case 41: return n / 128 * 18; case 143: return n / 128 * 28; case 8: return n / 32 * 34; case 2: return n / 32 * 18; }
  throw std::runtime_error("gguf: unsupported tensor type " + std::to_string(t));
}
void Gguf::open(const std::string& path) {
  base = h5_map_file(path, size);
  const uint8_t* p = (const uint8_t*)base;
  auto rd = [&](auto& v) { memcpy(&v, p, sizeof v); p += sizeof v; };
  auto str = [&]() { uint64_t n; rd(n); std::string s((const char*)p, n); p += n; return s; };
  uint32_t magic, ver; uint64_t nt, nk; rd(magic); rd(ver); rd(nt); rd(nk);
  if (magic != 0x46554747) throw std::runtime_error("not gguf");
  auto scalar = [&](uint32_t t, GValue& v) {
    switch (t) {
      case 0: { uint8_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 1: { int8_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 2: { uint16_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 3: { int16_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 4: { uint32_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 5: { int32_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 6: { float x; rd(x); v.nums.push_back(x); v.ints.push_back((int64_t)x); break; }
      case 7: { uint8_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 10: { uint64_t x; rd(x); v.ints.push_back((int64_t)x); v.nums.push_back((double)x); break; }
      case 11: { int64_t x; rd(x); v.ints.push_back(x); v.nums.push_back((double)x); break; }
      case 12: { double x; rd(x); v.nums.push_back(x); v.ints.push_back((int64_t)x); break; }
      case 8: v.strs.push_back(str()); break;
      default: throw std::runtime_error("gguf kv type");
    }
  };
  for (uint64_t i = 0; i < nk; i++) {
    std::string k = str(); GValue v; rd(v.type);
    if (v.type == 9) { rd(v.etype); uint64_t n; rd(n); for (uint64_t j = 0; j < n; j++) scalar(v.etype, v); }
    else scalar(v.type, v);
    kv[k] = std::move(v);
  }
  for (uint64_t i = 0; i < nt; i++) {
    GTensor t; t.name = str(); uint32_t nd; rd(nd); uint64_t n = 1;
    for (uint32_t j = 0; j < nd; j++) { uint64_t d; rd(d); t.ne.push_back(d); n *= d; }
    rd(t.type); rd(t.off); t.bytes = tbytes(t.type, n);
    index[t.name] = tensors.size(); tensors.push_back(t);
  }
  uint64_t align = 32; if (kv.count("general.alignment")) align = kv["general.alignment"].ints[0];
  uint64_t data = ((uint64_t)(p - (const uint8_t*)base) + align - 1) / align * align;
  for (auto& t : tensors) t.data = (const uint8_t*)base + data + t.off;
}
const GTensor& Gguf::t(const std::string& n) const {
  auto it = index.find(n); if (it == index.end()) throw std::runtime_error("missing tensor " + n);
  return tensors[it->second];
}
Gguf::~Gguf() { if (base) h5_unmap_file(base, size); }
}
