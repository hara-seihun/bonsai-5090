// nvcc -O2 -std=c++17 -arch=sm_120a -Isrc tools/sparse_head_probe.cu -o /tmp/sparse-head-probe
// /tmp/sparse-head-probe --host    (layout/packing checks, no CUDA device needed)
// /tmp/sparse-head-probe --device  (actual gather + bitwise checks + gather timing)
#include "sparse_head.cuh"
#include "compact.hpp"
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace h5;
static void require(bool ok, const char* message) {
  if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
static void ck(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }

struct Position { size_t word; int shift; unsigned mask; };
// Independent natural (row,K) encoder/decoder, not the production row gather.
static Position position(SparseHeadFormat f, int N, int K, int row, int k) {
  const int r = row % 32, kb = k / 128, x = k % 128;
  if (f == SparseHeadFormat::Int8Trits) {
    const int lane = (r % 8) * 4 + (x % 16) / 4;
    const int reg = (x / 32) * 8 + (r / 8) * 2 + (x % 32) / 16;
    return {(size_t(row / 32) * (K / 128) + kb) * 256 + lane * 8 + reg / 4,
            (reg % 4) * 2 + (x % 4) * 8, 3};
  }
  const bool nv = f == SparseHeadFormat::Nvfp4;
  const int lane = (r % 8) * 4 + (x % 32) / 8;
  const int word = nv ? (r / 16) * 8 + (x / 64) * 4 + ((x % 64) / 32) * 2 + (r / 8) % 2
                      : (r / 16) * 4 + (x / 64) * 2 + (r / 8) % 2;
  return {(size_t(kb) * (N / 32) + row / 32) * (nv ? 512 : 256) +
          (word / 4) * 128 + lane * 4 + word % 4,
          (x % 8) * 4, nv ? 15u : ((x / 32) % 2 ? 5u : 10u)};
}
static unsigned code(SparseHeadFormat f, const std::vector<uint32_t>& w,
                     int N, int K, int row, int k) {
  if (f == SparseHeadFormat::Dense5) {
    const int r = row % 32, x = k % 128;
    const int lane = (r % 8) * 4 + (x % 32) / 8;
    const int index = ((r / 16) * 4 + (x / 64) * 2 + (r / 8) % 2) * 16 +
                      ((x % 64) / 32) * 8 + x % 8;
    const int byte = index / 5;
    const size_t offset = byte < 24 ? (byte / 4) * 128 + lane * 4 + byte % 4
                                   : 768 + lane * 2 + byte - 24;
    unsigned packed = reinterpret_cast<const uint8_t*>(w.data())[
                      (size_t(k / 128) * (N / 32) + row / 32) * 832 + offset];
    for (int i = 0; i < index % 5; ++i) packed /= 3;
    return dense5::fp4(int(packed % 3) - 1);
  }
  const auto p = position(f, N, K, row, k);
  unsigned v = (w[p.word] >> p.shift) & p.mask;
  if (f == SparseHeadFormat::Fp4Trits && ((k / 32) % 2)) v <<= 1;
  return v;
}
static size_t scale_index(SparseHeadFormat f, int N, int row, int kb, int b = 0) {
  if (f == SparseHeadFormat::Int8Trits) return size_t(kb) * N + row;
  const size_t chunk = size_t(kb) * (N / 32) + row / 32;
  const int r = row % 32;
  return f == SparseHeadFormat::Nvfp4 ? chunk * 256 + r * 8 + b
       : chunk * 32 + (r % 8) * 4 + (r / 16) * 2 + (r / 8) % 2;
}
static int scale_width(SparseHeadFormat f) {
  return f == SparseHeadFormat::Int8Trits ? 4 : f == SparseHeadFormat::Nvfp4 ? 1 : 2;
}
static int scale_blocks(SparseHeadFormat f) { return f == SparseHeadFormat::Nvfp4 ? 8 : 1; }

struct DeviceBuffer {
  void* p;
  explicit DeviceBuffer(size_t bytes) { ck(cudaMalloc(&p, bytes)); }
  ~DeviceBuffer() { cudaFree(p); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
static void probe(SparseHeadFormat f, int count, bool device, int K, int alignment) {
  const int N = 96;
  SparseHeadLayout l;
  require(bool(sparse_head_layout(f, count, K, alignment, &l)), "layout");
  const int cw = f == SparseHeadFormat::Nvfp4 ? 2048 : f == SparseHeadFormat::Dense5 ? 832 : 1024;
  const int width = scale_width(f), blocks = scale_blocks(f);
  std::vector<uint32_t> source(size_t(N / 32) * (K / 128) * cw / 4, 0);
  std::vector<uint8_t> scales(size_t(N) * (K / 128) * width * blocks);
  std::vector<float> rs(N);
  for (int r = 0; r < N; ++r) {
    rs[r] = float(r + 1) / 1024.f;
    for (int k = 0; k < K && f != SparseHeadFormat::Dense5; ++k) {
      const auto p = position(f, N, K, r, k);
      const unsigned x = (r * 29 + k * 11 + (k / 7) * (r / 3)) % 3;
      unsigned v = f == SparseHeadFormat::Int8Trits ? (x == 2 ? 3 : x)
                 : f == SparseHeadFormat::Fp4Trits ? (x == 2 ? 10 : x * 2)
                 : (r * 13 + k * 7 + r * (k / 5)) % 16;
      if (f == SparseHeadFormat::Fp4Trits && ((k / 32) % 2)) v >>= 1;
      source[p.word] |= v << p.shift;
    }
    for (int kb = 0; kb < K / 128; ++kb)
      for (int b = 0; b < blocks; ++b) {
        const uint32_t value = f == SparseHeadFormat::Int8Trits ? 0x3e800000u + r * 8192 + kb * 64
                             : f != SparseHeadFormat::Nvfp4 ? 0x3000u + r * 8 + kb
                             : 1 + (r * 7 + kb * 13 + b * 3) % 120;
        memcpy(scales.data() + scale_index(f, N, r, kb, b) * width, &value, width);
      }
  }
  if (f == SparseHeadFormat::Dense5) {
    for (int kb = 0; kb < K / 128; ++kb) for (int g = 0; g < N / 32; ++g)
      for (int lane = 0; lane < 32; ++lane) {
        int8_t trits[128];
        for (int w = 0; w < 8; ++w) for (int j = 0; j < 16; ++j) {
          const int r = g * 32 + 16 * (w / 4) + 8 * (w % 2) + lane / 4;
          const int k = kb * 128 + 64 * ((w / 2) % 2) + 32 * (j / 8) + 8 * (lane % 4) + j % 8;
          trits[w * 16 + j] = int8_t((r * 29 + k * 11 + (k / 7) * (r / 3)) % 3 - 1);
        }
        dense5::pack_lane(reinterpret_cast<uint8_t*>(source.data()) +
                          (size_t(kb) * (N / 32) + g) * 832, lane, trits);
      }
  }
  std::vector<int> ids(count);
  for (int i = 0; i < count; ++i) ids[i] = (i * 37 + 95) % N;
  if (count > 2) ids[2] = ids[0];
  std::vector<uint32_t> output(l.weight_bytes / 4, ~0u);
  std::vector<uint8_t> os(l.scale_bytes, 0xff);
  std::vector<float> ors(l.rows, -1.f);
  std::vector<unsigned> status(l.groups, ~0u);
  if (device) {
    DeviceBuffer sw(source.size() * 4), ss(scales.size()), sr(N * 4), di(count * 4);
    DeviceBuffer dw(l.weight_bytes), ds(l.scale_bytes), dr(l.row_scale_bytes), st(l.status_bytes);
    ck(cudaMemcpy(sw.p, source.data(), source.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(ss.p, scales.data(), scales.size(), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(sr.p, rs.data(), N * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(di.p, ids.data(), count * 4, cudaMemcpyHostToDevice));
    SparseHeadSource src{sw.p, ss.p, static_cast<float*>(sr.p), N, K, f};
    SparseHeadDestination dst{dw.p, ds.p, static_cast<float*>(dr.p), static_cast<unsigned*>(st.p),
                              l.weight_bytes, l.scale_bytes, l.row_scale_bytes, l.status_bytes};
    const auto launch = [&] { const auto result = launch_sparse_head_gather(src, dst, static_cast<int*>(di.p), count, alignment, nullptr);
      if (!result) { fprintf(stderr, "gather error %d: %s\n", int(result.error), cudaGetErrorString(result.cuda_error)); exit(1); } };
    launch();
    ck(cudaMemcpy(output.data(), dw.p, l.weight_bytes, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(os.data(), ds.p, l.scale_bytes, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(ors.data(), dr.p, l.row_scale_bytes, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(status.data(), st.p, l.status_bytes, cudaMemcpyDeviceToHost));
    cudaEvent_t a, z; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&z));
    ck(cudaEventRecord(a)); for (int i = 0; i < 100; ++i) launch();
    ck(cudaEventRecord(z)); ck(cudaEventSynchronize(z));
    float ms; ck(cudaEventElapsedTime(&ms, a, z));
    ck(cudaEventDestroy(a)); ck(cudaEventDestroy(z));
    printf("device format=%d count=%d K=%d padded=%d gather_us=%.3f\n", int(f), count, K, l.rows, ms * 10.f);
    // Invalid IDs must never access source memory and must produce an explicit status.
    const int bad = N; ck(cudaMemcpy(di.p, &bad, sizeof(bad), cudaMemcpyHostToDevice));
    launch(); unsigned error = 0; ck(cudaMemcpy(&error, st.p, sizeof(error), cudaMemcpyDeviceToHost));
    require(error == 1, "invalid ID status");
  } else {
    for (int g = 0; g < l.groups; ++g) {
      int selected[32]; for (int r = 0; r < 32; ++r) selected[r] = g * 32 + r < count ? ids[g * 32 + r] : -1;
      status[g] = 0;
      for (int kb = 0; kb < l.kblocks; ++kb) {
        const size_t chunk = f == SparseHeadFormat::Int8Trits ? size_t(g) * l.kblocks + kb : size_t(kb) * l.groups + g;
        for (int word = 0; word < cw / 4; ++word)
          output[chunk * (cw / 4) + word] = sparse_head_detail::gather_word(f, source.data(), selected, N / 32, l.kblocks, kb, word);
        for (int r = 0; r < 32; ++r) for (int b = 0; b < blocks; ++b) {
          auto* d = os.data() + scale_index(f, l.rows, g * 32 + r, kb, b) * width;
          if (selected[r] < 0) memset(d, 0, width);
          else memcpy(d, scales.data() + scale_index(f, N, selected[r], kb, b) * width, width);
        }
      }
      for (int r = 0; r < 32; ++r) ors[g * 32 + r] = selected[r] < 0 ? 0.f : rs[selected[r]];
    }
  }
  for (unsigned s : status) require(s == 0, "status overwritten");
  for (int r = 0; r < l.rows; ++r) {
    for (int k = 0; k < K; ++k)
      require(code(f, output, l.rows, K, r, k) == (r < count ? code(f, source, N, K, ids[r], k) : 0), "selected/padded weight");
    for (int kb = 0; kb < K / 128; ++kb) for (int b = 0; b < blocks; ++b) {
      uint32_t actual = 0, expected = 0;
      memcpy(&actual, os.data() + scale_index(f, l.rows, r, kb, b) * width, width);
      if (r < count) memcpy(&expected, scales.data() + scale_index(f, N, ids[r], kb, b) * width, width);
      require(actual == expected, "selected/padded scale");
    }
    require(ors[r] == (r < count ? rs[ids[r]] : 0.f), "selected/padded row scale");
  }
}
static void selection_probe() {
  const int count = 5, ld = 32, rows = 2;
  const int ids[count] = {17, 3, 93, 8, 44};
  const int rowmap[4] = {0, 0, 1, 0};
  const float infinity = std::numeric_limits<float>::infinity();
  std::vector<float> logits(rows * ld, 1000.f);
  const float first[count] = {4.f, 4.f, -2.f, -infinity, 4.f};
  for (int i = 0; i < count; ++i) { logits[i] = first[i]; logits[ld + i] = -infinity; }
  const float delta = 10.f; int bits; memcpy(&bits, &delta, sizeof(bits));
  const int2 adj[2] = {make_int2(-1, 0), make_int2(93, bits)};
  DeviceBuffer dl(logits.size() * 4), di(sizeof(ids)), dm(sizeof(rowmap)), da(sizeof(adj));
  DeviceBuffer out(rows * 4), tv(rows * 16 * 4), ti(rows * 16 * 4);
  ck(cudaMemcpy(dl.p, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
  ck(cudaMemcpy(di.p, ids, sizeof(ids), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(dm.p, rowmap, sizeof(rowmap), cudaMemcpyHostToDevice));
  ck(cudaMemcpy(da.p, adj, sizeof(adj), cudaMemcpyHostToDevice));
  auto result = launch_sparse_argmax(static_cast<float*>(dl.p), ld, static_cast<int*>(di.p), count,
      static_cast<int*>(out.p), rows, static_cast<int2*>(da.p), 1, 2, static_cast<int*>(dm.p), nullptr);
  require(bool(result), "argmax launch");
  int winners[rows]; ck(cudaMemcpy(winners, out.p, sizeof(winners), cudaMemcpyDeviceToHost));
  require(winners[0] == 93 && winners[1] == 3, "adjusted argmax and all -inf legal tie");
  result = launch_sparse_topk(static_cast<float*>(dl.p), ld, static_cast<int*>(di.p), count,
      static_cast<float*>(tv.p), static_cast<int*>(ti.p), rows, nullptr);
  require(bool(result), "topk launch");
  float values[rows * 16]; int tokens[rows * 16];
  ck(cudaMemcpy(values, tv.p, sizeof(values), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(tokens, ti.p, sizeof(tokens), cudaMemcpyDeviceToHost));
  const int expected[4] = {3, 17, 44, 93};
  for (int i = 0; i < 4; ++i) require(tokens[i] == expected[i] && values[i] == (i < 3 ? 4.f : -2.f), "top16 tie order");
  for (int i = 4; i < rows * 16; ++i) require(tokens[i] == ids[0] && values[i] == -infinity, "top16 legal padding");
  puts("PASS: compact argmax adjustments, mapped original row, original-ID ties and safe top16 padding");
}
int main(int argc, char** argv) {
  require(argc == 2 && (!strcmp(argv[1], "--host") || !strcmp(argv[1], "--device")), "pass --host or --device");
  SparseHeadLayout l;
  require(sparse_head_layout(static_cast<SparseHeadFormat>(99), 1, 128, 32, &l).error == SparseHeadError::UnsupportedFormat, "unsupported format");
  require(sparse_head_layout(SparseHeadFormat::Fp4Trits, 0, 128, 32, &l).error == SparseHeadError::EmptyRows, "empty set");
  require(sparse_head_layout(SparseHeadFormat::Fp4Trits, 1, 127, 32, &l).error == SparseHeadError::InvalidShape, "unaligned K");
  for (auto f : {SparseHeadFormat::Int8Trits, SparseHeadFormat::Fp4Trits, SparseHeadFormat::Nvfp4, SparseHeadFormat::Dense5})
    for (int count : {1, 31, 32, 33, 65})
      probe(f, count, !strcmp(argv[1], "--device"), 384, f == SparseHeadFormat::Int8Trits ? 128 : 32);
  for (int count : {1, 31, 32, 33, 65})
    probe(SparseHeadFormat::Dense5, count, !strcmp(argv[1], "--device"), 5120, 128);
  if (!strcmp(argv[1], "--device")) selection_probe();
  puts("PASS: 25 gathers; row permutations, duplicate IDs, group crossings, padding and scale bits");
}
