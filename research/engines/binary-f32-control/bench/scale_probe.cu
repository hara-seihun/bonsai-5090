// Which lane/byte of the scale-A register scales which row / k16 block in
// mma.m16n8k64 kind::mxf4nvf4 scale_vec::4X? A = B = 1.0 only in k16 block J; scale-A of lane L
// is (L%16+1) or (L/16+1) in every byte (exact e4m3 integers), scale-B = 1.
#include <cstdio>
#include <cuda_fp8.h>
#include <cstdint>
__global__ void k(int J, int mode, float* out) {
  int lane = threadIdx.x, tq = lane & 3;
  // e2m1 1.0 = 0x2. A: a0,a1 cover k 8tq..8tq+7 (block tq/2), a2,a3 cover k 32+8tq.. (block 2+tq/2)
  unsigned one8 = 0x22222222u;
  unsigned a[4], b[2];
  a[0] = a[1] = ((tq >> 1) == J) ? one8 : 0; a[2] = a[3] = (2 + (tq >> 1) == J) ? one8 : 0;
  b[0] = ((tq >> 1) == J) ? one8 : 0; b[1] = (2 + (tq >> 1) == J) ? one8 : 0;
  float v = mode ? (lane / 16 + 1) : (lane % 16 + 1);
  unsigned char byte = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  unsigned sa = byte * 0x01010101u, sb = 0x38383838u; uint16_t z = 0;
  float d[4] = {0, 0, 0, 0};
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb), "h"(z));
  int gid = lane >> 2;
  out[gid * 8 + tq * 2] = d[0]; out[gid * 8 + tq * 2 + 1] = d[1]; out[(gid + 8) * 8 + tq * 2] = d[2]; out[(gid + 8) * 8 + tq * 2 + 1] = d[3];
}
int main() {
  float* o; cudaMallocManaged(&o, 128 * 4);
  for (int J = 0; J < 4; J++) {
    printf("k-block %d: row -> lane (scale/16 per mode):", J);
    for (int r = 0; r < 16; r++) {
      int lo, hi;
      k<<<1, 32>>>(J, 0, o); cudaDeviceSynchronize(); lo = (int)(o[r * 8] / 16 + 0.5f) - 1;
      k<<<1, 32>>>(J, 1, o); cudaDeviceSynchronize(); hi = (int)(o[r * 8] / 16 + 0.5f) - 1;
      printf(" %d:%d", r, hi * 16 + lo);
    }
    printf("\n");
  }
}
