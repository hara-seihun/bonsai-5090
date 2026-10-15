#include "gguf.h"
#include "kernels.cuh"
#include "compact.hpp"
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cstdio>
namespace h5 {
static inline void decode_ptq1(const uint8_t* blk, int8_t* trit, float* scale) {
  int e = 0;
  const int stages[2][2] = {{0, 16}, {16, 8}};
  for (int s = 0; s < 2; s++) {
    int j0 = stages[s][0], c = stages[s][1];
    for (int n = 0; n < 5; n++)
      for (int m = 0; m < c; m++) {
        unsigned q = blk[j0 + m];
        for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
        trit[e++] = (int8_t)((int)((q * 3u) >> 8) - 1);
      }
  }
  for (int n = 0; n < 4; n++)
    for (int h = 0; h < 2; h++) {
      unsigned q = blk[24 + h];
      for (int k = 0; k < n; k++) q = (q * 3u) & 0xff;
      trit[e++] = (int8_t)((int)((q * 3u) >> 8) - 1);
    }
  uint16_t hs; memcpy(&hs, blk + 26, 2);
  __half_raw r; r.x = hs; *scale = __half2float(__half(r));
}

static inline size_t sign_block_bytes(uint32_t type) {
  if (type == 41) return 18;
  if (type == 143) return 28;
  throw std::runtime_error("unsupported sign weight type " + std::to_string(type));
}
static inline void decode_sign_block(const uint8_t* blk, uint32_t type, int8_t* values, float* scale) {
  if (type == 143) { decode_ptq1(blk, values, scale); return; }
  uint16_t hs; memcpy(&hs, blk, 2);
  __half_raw r; r.x = hs; *scale = __half2float(__half(r));
  for (int i = 0; i < 128; i++) values[i] = ((blk[2 + i / 8] >> (i % 8)) & 1) ? 1 : -1;
}

static void repack_rows_fp4(const GTensor& t, int K, int Nt, int row0, std::vector<uint8_t>& W, std::vector<uint16_t>& S, int NGall, int nthreads, int gmul = 1, int goff = -1, bool compact = false, bool binary1 = false) {
  const int nkb = K / 128;
  std::atomic<int> next{0};
  auto worker = [&]() {
    int8_t trit[32][128]; float sc[32];
    for (;;) {
      int gl = next.fetch_add(1);
      if (gl >= Nt / 32) break;
      int g = goff >= 0 ? goff + gl * gmul : (row0 / 32) + gl;
      for (int kb = 0; kb < nkb; kb++) {
        for (int r = 0; r < 32; r++) decode_sign_block(t.data + ((size_t)(gl * 32 + r) * nkb + kb) * sign_block_bytes(t.type), t.type, trit[r], &sc[r]);
        uint8_t* chunk = W.data() + ((size_t)kb * NGall + g) * (compact ? 832 : binary1 ? 512 : 1024);
        for (int l = 0; l < 32; l++) {
          const int gid = l >> 2, tq = l & 3;
          if (binary1) {
            for (int w = 0; w < 8; w++) {
              const int m = w >> 2, h = (w >> 1) & 1, r = w & 1, row = 16 * m + 8 * r + gid;
              uint16_t word = 0;
              for (int j = 0; j < 8; j++) {
                word |= (uint16_t)(trit[row][64 * h + 8 * tq + j] < 0) << j;
                word |= (uint16_t)(trit[row][64 * h + 32 + 8 * tq + j] < 0) << (8 + j);
              }
              memcpy(chunk + l * 16 + w * 2, &word, 2);
            }
          } else if (compact) {
            int8_t stream[128];
            for (int w = 0; w < 8; w++) {
              const int m = w >> 2, h = (w >> 1) & 1, r = w & 1, row = 16 * m + 8 * r + gid;
              for (int j = 0; j < 8; j++) {
                stream[w * 16 + j] = trit[row][64 * h + 8 * tq + j];
                stream[w * 16 + 8 + j] = trit[row][64 * h + 32 + 8 * tq + j];
              }
            }
            dense5::pack_lane(chunk, l, stream);
          } else {
          for (int w = 0; w < 8; w++) {
            const int m = w >> 2, h = (w >> 1) & 1, r = w & 1, row = 16 * m + 8 * r + gid;
            uint32_t word = 0;
            for (int j = 0; j < 8; j++) {
              const int t0 = trit[row][64 * h + 8 * tq + j], t1 = trit[row][64 * h + 32 + 8 * tq + j];
              word |= (uint32_t)((t0 != 0) << 1 | (t0 < 0) << 3) << (4 * j);
              word |= (uint32_t)((t1 != 0) << 0 | (t1 < 0) << 2) << (4 * j);
            }
            memcpy(chunk + (w >> 2) * 512 + l * 16 + (w & 3) * 4, &word, 4);
          }
          }
        }
        uint16_t* sh = S.data() + ((size_t)kb * NGall + g) * 32;
        for (int gid = 0; gid < 8; gid++)
          for (int q = 0; q < 4; q++) {
            const int m = q >> 1, r = q & 1, row = 16 * m + 8 * r + gid;
            __half hv = __float2half(sc[row]); uint16_t u; memcpy(&u, &hv, 2); sh[gid * 4 + q] = u;
          }
      }
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
  for (auto& x : th) x.join();
}

}

#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return 1; } } while(0)
int main(int argc,char**argv) {
 if(argc!=2) return 2;
 h5::Gguf g;g.open(argv[1]);
 struct Case { std::string name; std::vector<h5::GTensor> tensors; std::vector<uint8_t> joined; bool il; };
 std::vector<Case> tests;
 for(const char* n:{"blk.0.attn_qkv.weight","blk.0.ssm_out.weight","blk.0.ffn_down.weight","output.weight"}) {
  auto t=g.t(n);t.ne[1]=64;t.bytes=64*(t.ne[0]/128)*18;tests.push_back({n,{t},{},false});
 }
 Case gu{"gate-up",{g.t("blk.0.ffn_gate.weight"),g.t("blk.0.ffn_up.weight")},{},true};
 for(auto&t:gu.tensors){t.ne[1]=64;t.bytes=64*(t.ne[0]/128)*18;}tests.push_back(std::move(gu));
 Case ab{"alpha-beta",{g.t("blk.0.ssm_alpha.weight")},{},false};
 auto b=g.t("blk.0.ssm_beta.weight");auto&t=ab.tensors[0];ab.joined.resize(t.bytes+b.bytes);memcpy(ab.joined.data(),t.data,t.bytes);memcpy(ab.joined.data()+t.bytes,b.data,b.bytes);t.ne[1]=96;t.bytes=ab.joined.size();t.data=ab.joined.data();tests.push_back(std::move(ab));
 for(auto&tc:tests) for(bool binary1:{false,true}) {
  int N=0,K=tc.tensors[0].ne[0];for(auto&t:tc.tensors)N+=t.ne[1];
  size_t nb=size_t(N/32)*(K/128)*(binary1?512:1024),ns=size_t(N)*(K/128);
  std::vector<uint8_t> cpu(nb,0),gpu(nb);std::vector<uint16_t>cs(ns,0),gs(ns);
  uint8_t*dw;uint16_t*ds;CHECK(cudaMalloc(&dw,nb));CHECK(cudaMalloc(&ds,ns*2));CHECK(cudaMemset(dw,0,nb));CHECK(cudaMemset(ds,0,ns*2));
  int row0=0,idx=0;
  for(auto&t:tc.tensors) {
   h5::repack_rows_fp4(t,K,t.ne[1],row0,cpu,cs,N/32,2,tc.il?2:1,tc.il?idx:-1,false,binary1);
   uint8_t*dr;CHECK(cudaMalloc(&dr,t.bytes));CHECK(cudaMemcpy(dr,t.data,t.bytes,cudaMemcpyHostToDevice));
   CHECK(h5::launch_repack_q1_fp4(dr,K,t.ne[1],dw,ds,N/32,tc.il?2:1,tc.il?idx:row0/32,binary1,0));
   CHECK(cudaDeviceSynchronize());CHECK(cudaFree(dr));row0+=t.ne[1];idx++;
  }
  CHECK(cudaMemcpy(gpu.data(),dw,nb,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gs.data(),ds,ns*2,cudaMemcpyDeviceToHost));CHECK(cudaFree(dw));CHECK(cudaFree(ds));
  size_t nw=0,nsbad=0;for(size_t i=0;i<nb;i++)nw+=cpu[i]!=gpu[i];for(size_t i=0;i<ns;i++)nsbad+=cs[i]!=gs[i];
  printf("%s binary1=%d N=%d K=%d: weight_bytes=%zu scale_bytes=%zu mismatches=%zu/%zu\n",tc.name.c_str(),int(binary1),N,K,nb,ns*2,nw,nsbad);
  if(nw||nsbad)return 1;
 }
 return 0;
}
