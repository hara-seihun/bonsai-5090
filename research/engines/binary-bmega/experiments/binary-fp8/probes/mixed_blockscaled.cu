__global__ void probe(float* p){float c[4]={};unsigned a=0,b=0,scale=0x7f7f7f7f;unsigned short z=0;
asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3},%6,{%7,%7},%6,{%7,%7};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a),"r"(b),"r"(scale),"h"(z));p[threadIdx.x]=c[0];}
