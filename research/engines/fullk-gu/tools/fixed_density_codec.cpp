// CPU-only exact 86-of-128 trit codec; see FIXED_DENSITY_CODEC.md for the wire contract.
#include "compact.hpp"
#include "gguf.h"
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <string>
#include <variant>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

namespace fd {
using U = unsigned __int128;
using Trits = std::array<int8_t, 128>;
using Packed = std::array<uint8_t, 25>;
using Original = std::array<uint8_t, 28>;
enum class Error { InvalidTrit, WrongDensity, InvalidMask, InvalidRank, InvalidSigns,
                   InvalidExtent, OutOfRange, BrokenStream, GgufOpen };
template<class T> using Result = std::variant<T, Error>;
struct Bits { U zero, negative; bool operator==(const Bits&) const = default; };
struct Block { Trits trits; uint16_t scale; bool operator==(const Block&) const = default; };
enum class Search { Linear, Binary };
constexpr U rank_mask = (U(1) << 114) - 1;
constexpr unsigned stride = 512;
struct Table {
  std::array<std::array<U, 43>, 129> c{};
  Table() {
    c[0][0] = 1;
    for (unsigned n = 1; n <= 128; ++n) {
      c[n][0] = 1;
      for (unsigned k = 1; k <= 42; ++k) c[n][k] = c[n-1][k] + c[n-1][k-1];
    }
  }
};
unsigned pop(U x) { return std::popcount(uint64_t(x)) + std::popcount(uint64_t(x >> 64)); }
unsigned ctz(U x) {
  return uint64_t(x) ? std::countr_zero(uint64_t(x)) : 64 + std::countr_zero(uint64_t(x >> 64));
}
Result<Bits> bits(const Trits& t) {
  Bits b{0, 0};
#if defined(__AVX2__)
  const auto zero = _mm256_setzero_si256(), minus_one = _mm256_set1_epi8(-1), one = _mm256_set1_epi8(1);
  for (unsigned i = 0; i < 128; i += 32) {
    auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(t.data()+i));
    auto invalid = _mm256_or_si256(_mm256_cmpgt_epi8(minus_one,v), _mm256_cmpgt_epi8(v,one));
    if (_mm256_movemask_epi8(invalid)) return Error::InvalidTrit;
    b.zero |= U(uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v,zero)))) << i;
    b.negative |= U(uint32_t(_mm256_movemask_epi8(v))) << i;
  }
#else
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t zero = 0, negative = 0;
    for (unsigned i = 0; i < 64; ++i) {
      auto v = t[half*64+i];
      if (v < -1 || v > 1) return Error::InvalidTrit;
      zero |= uint64_t(v == 0) << i; negative |= uint64_t(v == -1) << i;
    }
    b.zero |= U(zero) << (64*half); b.negative |= U(negative) << (64*half);
  }
#endif
  return b;
}
Trits trits(Bits b) {
  Trits t;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t zero = uint64_t(b.zero >> (64*half)), negative = uint64_t(b.negative >> (64*half));
#if defined(__AVX512BW__)
    auto positive = _mm512_maskz_set1_epi8(~(zero | negative), 1);
    auto minus_one = _mm512_maskz_set1_epi8(negative, -1);
    _mm512_storeu_si512(t.data()+64*half, _mm512_or_si512(positive,minus_one));
#else
    for (unsigned i = 0; i < 64; ++i) t[64*half+i] = 1-((zero>>i)&1)-2*((negative>>i)&1);
#endif
  }
  return t;
}
Result<U> rank(const Table& a, U mask, unsigned n, unsigned k) {
  if (n > 128 || k > 42 || k > n || (n < 128 && (mask >> n)) || pop(mask) != k)
    return Error::InvalidMask;
  U r = 0;
  for (unsigned j = 1; mask; ++j) {
    unsigned p = ctz(mask); mask &= mask - 1; r += a.c[p][j];
  }
  return r;
}
Result<U> unrank(const Table& a, U r, unsigned n, unsigned k, Search search) {
  if (n > 128 || k > 42 || k > n || r >= a.c[n][k]) return Error::InvalidRank;
  U mask = 0;
  if (search == Search::Linear) {
    for (int p = int(n) - 1; k; --p) {
      if (r >= a.c[p][k]) { r -= a.c[p][k]; mask |= U(1) << p; --k; }
    }
  } else {
    unsigned upper = n;
    while (k) {
      unsigned lo = k - 1, hi = upper;
      while (lo + 1 < hi) {
        unsigned p = (lo + hi) / 2;
        if (a.c[p][k] <= r) lo = p; else hi = p;
      }
      r -= a.c[lo][k]; mask |= U(1) << lo; upper = lo; --k;
    }
  }
  return mask;
}
U gather_signs(Bits b) {
#if defined(__BMI2__)
  uint64_t low_nz = ~uint64_t(b.zero), high_nz = ~uint64_t(b.zero >> 64);
  return U(_pext_u64(uint64_t(b.negative), low_nz)) |
    (U(_pext_u64(uint64_t(b.negative >> 64), high_nz)) << std::popcount(low_nz));
#else
  U s = 0; unsigned j = 0;
  for (unsigned i = 0; i < 128; ++i) if (!((b.zero >> i) & 1))
    s |= ((b.negative >> i) & 1) << j++;
  return s;
#endif
}
U scatter_signs(U zero, U s) {
#if defined(__BMI2__)
  uint64_t low_nz = ~uint64_t(zero), high_nz = ~uint64_t(zero >> 64);
  return U(_pdep_u64(uint64_t(s), low_nz)) |
    (U(_pdep_u64(uint64_t(s >> std::popcount(low_nz)), high_nz)) << 64);
#else
  U negative = 0; unsigned j = 0;
  for (unsigned i = 0; i < 128; ++i) if (!((zero >> i) & 1))
    negative |= ((s >> j++) & 1) << i;
  return negative;
#endif
}
Result<Packed> encode(const Table& a, Bits b) {
  if (b.zero & b.negative) return Error::InvalidSigns;
  if (pop(b.zero) != 42) return Error::WrongDensity;
  auto rr = rank(a, b.zero, 128, 42);
  if (auto e = std::get_if<Error>(&rr)) return *e;
  U s = gather_signs(b), low = std::get<U>(rr) | (s << 114), high = s >> 14;
  Packed p;
  for (unsigned i = 0; i < 16; ++i) p[i] = uint8_t(low >> (8*i));
  for (unsigned i = 0; i < 9; ++i) p[16+i] = uint8_t(high >> (8*i));
  return p;
}
Result<Bits> decode(const Table& a, const Packed& p, Search search) {
  U low = 0, high = 0;
  for (unsigned i = 0; i < 16; ++i) low |= U(p[i]) << (8*i);
  for (unsigned i = 0; i < 9; ++i) high |= U(p[16+i]) << (8*i);
  auto zz = unrank(a, low & rank_mask, 128, 42, search);
  if (auto e = std::get_if<Error>(&zz)) return *e;
  U zero = std::get<U>(zz), s = (low >> 114) | (high << 14);
  return Bits{zero, scatter_signs(zero, s)};
}
Block source(const Original& raw) {
  Block b; dense5::decode_ptq(raw.data(), b.trits.data(), b.scale); return b;
}
Result<Packed> encode(const Table& a, const Trits& t) {
  auto b = bits(t); if (auto e = std::get_if<Error>(&b)) return *e;
  return encode(a, std::get<Bits>(b));
}

// Constructed streams retain explicit original-byte escapes, not a trit re-encode.
class Stream {
  size_t n;
  std::vector<uint8_t> payload;
  std::vector<uint64_t> escape, prefix;
  std::vector<uint16_t> scales;
  Stream(size_t count) : n(count), escape((count+63)/64), prefix((count+511)/512+1), scales(count) {}
  bool flag(size_t i) const { return (escape[i/64] >> (i%64)) & 1; }
public:
  static Result<Stream> build(const Table& a, const std::vector<Original>& raw) {
    Stream out(raw.size()); uint64_t escapes = 0;
    out.payload.reserve(raw.size()*26);
    for (size_t i = 0; i < raw.size(); ++i) {
      if (i % stride == 0) out.prefix[i/stride] = escapes;
      auto b = source(raw[i]); out.scales[i] = b.scale;
      auto r = encode(a, b.trits);
      if (auto p = std::get_if<Packed>(&r)) out.payload.insert(out.payload.end(), p->begin(), p->end());
      else if (std::get<Error>(r) == Error::WrongDensity) {
        out.escape[i/64] |= uint64_t(1) << (i%64); ++escapes;
        out.payload.insert(out.payload.end(), raw[i].begin(), raw[i].begin()+26);
      } else return std::get<Error>(r);
    }
    out.prefix.back() = escapes;
    return out;
  }
  Result<size_t> address(size_t i) const {
    if (i >= n) return Error::OutOfRange;
    uint64_t before = prefix[i/stride];
    for (size_t w = (i/stride)*8; w < i/64; ++w) before += std::popcount(escape[w]);
    before += std::popcount(escape[i/64] & ((uint64_t(1) << (i%64))-1));
    return 25*i + before;
  }
  Result<bool> is_escape(size_t i) const {
    if (i >= n) return Error::OutOfRange;
    return flag(i);
  }
  Result<Block> at(const Table& a, size_t i, Search search) const {
    auto addr = address(i); if (auto e = std::get_if<Error>(&addr)) return *e;
    size_t off = std::get<size_t>(addr), length = flag(i) ? 26 : 25;
    if (off > payload.size() || length > payload.size()-off) return Error::BrokenStream;
    if (length == 26) {
      Original raw; std::memcpy(raw.data(), payload.data()+off, 26);
      std::memcpy(raw.data()+26, &scales[i], 2); return source(raw);
    }
    Packed p; std::memcpy(p.data(), payload.data()+off, 25);
    auto b = decode(a, p, search); if (auto e = std::get_if<Error>(&b)) return *e;
    return Block{trits(std::get<Bits>(b)), scales[i]};
  }
  Result<bool> escape_equals(size_t i, const Original& raw) const {
    auto addr = address(i); if (auto e = std::get_if<Error>(&addr)) return *e;
    if (!flag(i)) return Error::WrongDensity;
    return std::memcmp(payload.data()+std::get<size_t>(addr), raw.data(), 26) == 0;
  }
  size_t payload_bytes() const { return payload.size(); }
  size_t bitmap_bytes() const { return escape.size()*8; }
  size_t index_bytes() const { return prefix.size()*8; }
  uint64_t escapes() const { return prefix.back(); }
};
}
using namespace fd;
using cpp_int = boost::multiprecision::uint256_t;
void check(bool ok, const char* label) {
  if (!ok) { std::fprintf(stderr, "counterexample: %s\n", label); std::exit(1); }
}
template<class T> T require(Result<T> r, const char* label) {
  if (auto p = std::get_if<T>(&r)) return std::move(*p);
  std::fprintf(stderr, "counterexample: %s error=%d\n", label, int(std::get<Error>(r))); std::exit(1);
}
cpp_int oracle_choose(unsigned n, unsigned k) {
  if (k > n) return 0;
  cpp_int r = 1;
  for (unsigned j = 1; j <= k; ++j) { r *= n-j+1; r /= j; }
  return r;
}
cpp_int big(U x) { return (cpp_int(uint64_t(x>>64)) << 64) | uint64_t(x); }
uint64_t rng(uint64_t& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
Block reference_source(const Original& raw) {
  Block out; unsigned at = 0;
  for (unsigned stage = 0; stage < 3; ++stage) {
    unsigned off = stage == 0 ? 0 : stage == 1 ? 16 : 24;
    unsigned count = stage == 0 ? 16 : stage == 1 ? 8 : 2;
    unsigned digits = stage == 2 ? 4 : 5, power = 3;
    for (unsigned j = 0; j < digits; ++j, power *= 3)
      for (unsigned k = 0; k < count; ++k) out.trits[at++] = ((raw[off+k]*power/256)%3)-1;
  }
  std::memcpy(&out.scale, raw.data()+26, 2); return out;
}
void tests(const Table& a) {
  check(oracle_choose(128,42) == cpp_int("11328711813884840440197243767409600"), "independent binomial width");
  check(big(a.c[128][42]) == oracle_choose(128,42), "table width");
  check(a.c[128][42] > (U(1)<<113) && a.c[128][42] <= (U(1)<<114), "114 bits necessary/sufficient");
  for (unsigned n = 0; n <= 128; ++n) for (unsigned k = 0; k <= 42; ++k)
    check(big(a.c[n][k]) == oracle_choose(n,k), "Pascal vs independent multiplicative 256-bit oracle");
  unsigned small = 0;
  for (unsigned n = 0; n <= 12; ++n) {
    std::array<unsigned,13> next{};
    for (unsigned m = 0; m < (1u<<n); ++m) {
      unsigned k = std::popcount(m); U r = require(rank(a,m,n,k), "small rank");
      check(r == next[k]++, "colex equals enumerated sorted masks");
      for (auto route : {Search::Linear, Search::Binary})
        check(require(unrank(a,r,n,k,route), "small unrank") == m, "exhaustive small inverse");
      ++small;
    }
  }
  uint64_t seed = 0x50c9e619e368b17aULL;
  std::vector<U> masks{(U(1)<<42)-1, ((U(1)<<42)-1)<<86};
  U alternating = 0; for (unsigned i = 0; i < 42; ++i) alternating |= U(1)<<(i*3); masks.push_back(alternating);
  for (unsigned t = 0; t < 4096; ++t) {
    U z = 0; while (pop(z) < 42) z |= U(1) << (rng(seed)%128); masks.push_back(z);
  }
  for (size_t i = 0; i < masks.size(); ++i) {
    U z = masks[i], r = require(rank(a,z,128,42), "mask rank");
    cpp_int reference = 0; unsigned j = 0;
    for (unsigned p = 0; p < 128; ++p) if ((z>>p)&1) reference += oracle_choose(p,++j);
    check(big(r) == reference, "independent rank oracle");
    if (i == 0) check(r == 0, "minimum rank");
    if (i == 1) check(r == a.c[128][42]-1, "maximum rank");
    U random_signs = U(rng(seed)) | (U(rng(seed))<<64);
    for (U neg : {U(0), ~z, random_signs & ~z}) {
      Bits b{z,neg}; Packed packed = require(encode(a,b), "sign encode");
      cpp_int wire = reference, sign = 0; unsigned sign_index = 0;
      for (unsigned pos = 0; pos < 128; ++pos) if (!((z>>pos)&1)) {
        if ((neg>>pos)&1) sign |= cpp_int(1)<<sign_index;
        ++sign_index;
      }
      wire |= sign<<114; cpp_int actual = 0;
      for (unsigned byte = 0; byte < 25; ++byte) actual |= cpp_int(packed[byte])<<(8*byte);
      check(actual == wire, "independent exact wire bits/sign order");
      for (auto route : {Search::Linear, Search::Binary}) {
        Bits got = require(decode(a,packed,route), "sign decode");
        check(got == b, "extreme/random signs inverse");
        check(require(encode(a,trits(got)), "trit encode") == packed, "canonical wire re-encode");
      }
    }
  }
  Trits invalid{}; invalid[0] = 2;
  check(std::get<Error>(encode(a,invalid)) == Error::InvalidTrit, "invalid trit rejected");
  check(std::get<Error>(encode(a,Trits{})) == Error::WrongDensity, "non86 explicit escape eligibility");
  check(std::get<Error>(encode(a,Bits{masks[0],1})) == Error::InvalidSigns, "overlapping signs rejected");
  for (auto route : {Search::Linear, Search::Binary}) {
    Packed bad{}; U r = a.c[128][42];
    for (unsigned i = 0; i < 15; ++i) bad[i] = uint8_t(r>>(8*i));
    check(std::get<Error>(decode(a,bad,route)) == Error::InvalidRank, "unused rank rejected");
    bad.fill(255); check(std::get<Error>(decode(a,bad,route)) == Error::InvalidRank, "all-one invalid rank");
  }
  for (unsigned byte = 0; byte < 256; ++byte) {
    Original raw; raw.fill(byte);
    check(source(raw) == reference_source(raw), "all PTQ bytes independent source decoder");
  }
  Original fixed{}; std::fill(fixed.begin(),fixed.begin()+8,128);
  unsigned tail = 0;
  for (; tail < 256; ++tail) {
    fixed[24] = uint8_t(tail);
    if (pop(require(bits(source(fixed).trits), "synthetic fixed bits").zero) == 42) break;
  }
  check(tail < 256, "synthetic fixed PTQ source exists");
  Original alias = fixed; alias[8] = 1;
  check(alias != fixed && source(alias) == source(fixed), "noninjective original-byte counterexample");
  check(require(encode(a,source(alias).trits), "alias encode") == require(encode(a,source(fixed).trits), "fixed encode"),
    "codec preserves weight identity rather than upstream byte aliases");
  std::vector<Original> artificial(1100);
  for (size_t i = 0; i < artificial.size(); ++i) {
    if (i%3 == 2) artificial[i] = fixed;
    else artificial[i].fill(i%3 == 0 ? 0 : 128);
    artificial[i][26] = uint8_t(i); artificial[i][27] = uint8_t(i>>8);
  }
  auto stream = require(Stream::build(a,artificial), "synthetic stream");
  check(stream.escapes() == 734, "synthetic escape and fixed counts");
  size_t off = 0;
  for (size_t i = 0; i < artificial.size(); ++i) {
    check(require(stream.address(i), "synthetic address") == off, "prefix/bitmap vs sequential address");
    check(require(stream.at(a,i,Search::Linear), "synthetic read") == source(artificial[i]), "synthetic escape/scale identity");
    bool escaped = require(stream.is_escape(i), "synthetic flag");
    if (escaped) check(require(stream.escape_equals(i,artificial[i]), "escape bytes"), "byte-exact escape");
    off += escaped ? 26 : 25;
  }
  auto empty = require(Stream::build(a,{}), "empty stream");
  check(std::get<Error>(empty.address(0)) == Error::OutOfRange, "empty stream address");
  check(std::get<Error>(stream.address(1100)) == Error::OutOfRange, "past-end address");
  std::printf("tests small_masks=%u full_masks=%zu sign_cases=%zu all_ptq_bytes=256 status=pass\n", small, masks.size(), masks.size()*3);
}
Result<std::vector<Original>> samples(const std::string& path) {
  h5::Gguf gg;
  try { gg.open(path); } catch (const std::exception& e) {
    std::fprintf(stderr,"gguf boundary: %s\n", e.what()); return Error::GgufOpen;
  }
  std::vector<Original> raw; size_t tensors = 0, inventory = 0;
  for (const auto& t : gg.tensors) if (t.type == 143 && t.ne.size() == 2) {
    if (t.ne[0]%128 || t.bytes != t.ne[0]*t.ne[1]/128*28 ||
        t.data < static_cast<uint8_t*>(gg.base) ||
        size_t(t.data-static_cast<uint8_t*>(gg.base)) > gg.size ||
        t.bytes > gg.size-size_t(t.data-static_cast<uint8_t*>(gg.base))) return Error::InvalidExtent;
    size_t count = t.bytes/28, take = std::min(size_t(64),count);
    ++tensors; inventory += count;
    for (size_t i = 0; i < take; ++i) {
      size_t block = take == 1 ? 0 : i*(count-1)/(take-1);
      Original b; std::memcpy(b.data(),t.data+block*28,28); raw.push_back(b);
    }
  }
  if (raw.empty()) return Error::InvalidExtent;
  std::printf("source tensors=%zu inventory_blocks=%zu samples=%zu sampling=64_evenly_spaced_per_matrix\n", tensors,inventory,raw.size());
  return raw;
}
template<class F> void bench(const char* name, size_t count, unsigned repeats, F fn) {
  auto start = std::chrono::steady_clock::now();
  for (unsigned r = 0; r < repeats; ++r) for (size_t i = 0; i < count; ++i) {
    auto value = fn(i);
    asm volatile("" : : "m"(value) : "memory");
  }
  double ns = std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/(count*repeats);
  std::printf("bench name=%s blocks=%zu repeats=%u ns_per_block=%.3f blocks_per_s=%.3f\n",name,count,repeats,ns,1e9/ns);
}
int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr,"usage: fixed-density-codec MODEL.gguf\n"); return 2; }
  Table a; tests(a);
  auto sr = samples(argv[1]);
  if (auto e = std::get_if<Error>(&sr)) { std::fprintf(stderr,"sample_error=%d\n",int(*e)); return 1; }
  auto raw = std::get<std::vector<Original>>(std::move(sr));
  auto stream = require(Stream::build(a,raw), "real stream");
  std::vector<Bits> b; std::vector<Trits> t; std::vector<Packed> p; std::vector<U> ranks;
  size_t off = 0;
  for (size_t i = 0; i < raw.size(); ++i) {
    Block original = source(raw[i]); check(original == reference_source(raw[i]), "real independent PTQ decoder");
    check(require(stream.address(i), "real address") == off, "real address vs running cursor");
    for (auto route : {Search::Linear, Search::Binary})
      check(require(stream.at(a,i,route), "real stream decode") == original, "real trits/scales exact");
    bool escaped = require(stream.is_escape(i), "real flag");
    if (escaped) check(require(stream.escape_equals(i,raw[i]), "real escape bytes"), "real original bytes preserved");
    else {
      t.push_back(original.trits); b.push_back(require(bits(t.back()), "real masks"));
      p.push_back(require(encode(a,b.back()), "real encode"));
      ranks.push_back(require(rank(a,b.back().zero,128,42), "real rank"));
    }
    off += escaped ? 26 : 25;
  }
  check(off == stream.payload_bytes(), "exact payload extent");
  size_t all = stream.payload_bytes()+stream.bitmap_bytes()+stream.index_bytes()+raw.size()*2;
  std::printf("stream blocks=%zu fixed=%zu escapes=%llu payload_bytes=%zu bitmap_bytes=%zu index_bytes=%zu scale_bytes=%zu total_bytes=%zu original_28_bytes=%zu saved_bytes=%lld\n",
    raw.size(),b.size(),(unsigned long long)stream.escapes(),stream.payload_bytes(),stream.bitmap_bytes(),stream.index_bytes(),raw.size()*2,all,raw.size()*28,(long long)(raw.size()*28)-(long long)all);
  std::printf("ingredients rank_bits=114 signs=86 table_bytes=%zu table_entries=%zu index_stride=512 bitmap_word_bits=64 max_address_popcounts=8 bmi2=%d\n",
    sizeof(a),a.c.size()*a.c[0].size(),
#if defined(__BMI2__)
    1
#else
    0
#endif
  );
  if (b.empty()) { std::fprintf(stderr,"no fixed86 samples for benchmark\n"); return 1; }
  constexpr unsigned repeats = 4;
  bench("source_ptq_to_trits",raw.size(),repeats,[&](size_t i){return source(raw[i]);});
  bench("rank_zero_mask",b.size(),repeats,[&](size_t i){return rank(a,b[i].zero,128,42);});
  bench("unrank_linear",b.size(),repeats,[&](size_t i){return unrank(a,ranks[i],128,42,Search::Linear);});
  bench("unrank_binary",b.size(),repeats,[&](size_t i){return unrank(a,ranks[i],128,42,Search::Binary);});
  bench("trits_to_masks",t.size(),repeats,[&](size_t i){return bits(t[i]);});
  bench("masks_to_trits",b.size(),repeats,[&](size_t i){return trits(b[i]);});
  bench("gather_signs",b.size(),repeats,[&](size_t i){return gather_signs(b[i]);});
  bench("encode_masks_to_25B",b.size(),repeats,[&](size_t i){return encode(a,b[i]);});
  bench("encode_trits_to_25B",t.size(),repeats,[&](size_t i){return encode(a,t[i]);});
  bench("decode_25B_to_masks_linear",p.size(),repeats,[&](size_t i){return decode(a,p[i],Search::Linear);});
  bench("decode_25B_to_masks_binary",p.size(),repeats,[&](size_t i){return decode(a,p[i],Search::Binary);});
  bench("decode_25B_to_trits_linear",p.size(),repeats,[&](size_t i){return trits(require(decode(a,p[i],Search::Linear),"bench decode"));});
  bench("address_lookup",raw.size(),repeats,[&](size_t i){return stream.address((i*7919)%raw.size());});
  bench("stream_random_read_linear",raw.size(),repeats,[&](size_t i){return stream.at(a,(i*7919)%raw.size(),Search::Linear);});
  std::puts("result exact_trits_and_scale=pass byte_exact_escapes=pass scope=bounded_samples cpu_only=true");
  return std::ferror(stdout) ? 1 : 0;
}
