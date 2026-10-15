#!/usr/bin/env python3
"""Generate bounded sparse sidecar from immutable measured dense ring source."""
from pathlib import Path
from sparse_third import recode
E=Path(__file__).resolve().parent
src=(E/'signed_digits.cu').read_text()
assert 'namespace signed_fp4 {' in src
src=src.replace('#include "signed_digits.cuh"','#include "sparse_digits.cuh"').replace('namespace signed_fp4 {','namespace sparse_fp4 {')
start=src.index('template<bool PLANES>')
end=src.index('using namespace h5;',start)
src=src[:start]+src[end:]
src=src.replace('64*k*3/2','64*k')
pos=src.index('size_t partial_floats')
src=src[:pos]+'''size_t count_bytes(int k,int r){return digit_bytes(k,r)/128;}
size_t entry_words(int k,int r){return count_bytes(k,r)*ListCapacity;}
'''+src[pos:]
src=src.replace('(p.layout!=WeightLayout::WF8&&p.layout!=WeightLayout::Packed1)','p.layout!=WeightLayout::Packed1')
start=src.index('__device__ __forceinline__ unsigned code(')
end=src.index('__device__ __forceinline__ unsigned expand(',start)
D={0:0,1:1,2:2,3:3,4:4,6:5,8:6,12:7}
def code(v):return D[abs(v)]|(8 if v<0 else 0)
vals=[]
for q in range(-128,128):
 a,b,c=recode[q]
 assert q==a+4*b+16*c
 vals.append(code(a)|(code(b)<<4)|((c&255)<<8))
lookup='__device__ const uint16_t labels[256]={'+','.join(str(v) for v in vals)+'};\n'
pack='''
__global__ void pack(bitserial::Carrier carrier,Digits out,int k,int rows){
  int kb=blockIdx.x,t=blockIdx.y*64+threadIdx.x/4,c=threadIdx.x%4;
  size_t base=(size_t(blockIdx.y)*(k/128)+kb)*64+threadIdx.x/4;
  size_t plane=size_t((rows+63)/64)*64*k/2;
  uint32_t words[2][4]={};unsigned mask=0;
#pragma unroll
  for(int j=0;j<32;j++){
    int v=t<rows?int(carrier.q[size_t(t)*k+kb*128+c*32+j]):0;
    unsigned label=labels[v+128];mask|=unsigned((label>>8)!=0)<<j;
#pragma unroll
    for(int p=0;p<2;p++)words[p][j/8]|=((label>>(p*4))&15)<<(4*(j%8));
  }
#pragma unroll
  for(int p=0;p<2;p++)*(uint4*)(out.data+size_t(p)*plane+base*64+((c^(((threadIdx.x/4)>>1)&3))<<4))=make_uint4(words[p][0],words[p][1],words[p][2],words[p][3]);
  int prefix=__popc(mask);
  int x=__shfl_up_sync(0xffffffff,prefix,1);if(c>=1)prefix+=x;
  x=__shfl_up_sync(0xffffffff,prefix,2);if(c>=2)prefix+=x;
  int total=__shfl_sync(0xffffffff,prefix,(threadIdx.x&31)/4*4+3);
  if(c==0)out.counts[base]=uint8_t(total);
  if(t<rows&&total<=ListCapacity){
    int dst=prefix-__popc(mask);
    while(mask){int j=__ffs(mask)-1;mask&=mask-1;
      unsigned label=labels[int(carrier.q[size_t(t)*k+kb*128+c*32+j])+128];
      out.entries[base*ListCapacity+dst++]=uint16_t((label&0xff00)|(c*32+j));
    }
  }
}
cudaError_t encode(bitserial::Carrier c,int k,int r,Digits d,cudaStream_t st){
  size_t need=digit_bytes(k,r),counts=count_bytes(k,r),entries=entry_words(k,r);
  if(!need||r>65535||!c.q||!c.scales||!c.status||!d.data||d.capacity<need||uintptr_t(d.data)%16||
     !d.counts||d.count_capacity<counts||!d.entries||d.entry_capacity<entries||uintptr_t(d.entries)%2)return cudaErrorInvalidValue;
  pack<<<dim3(k/128,(r+63)/64),256,0,st>>>(c,d,k,r);return cudaGetLastError();
}
'''
src=src[:start]+lookup+pack+src[end:]
src=src.replace('STAGE=SW+SS+3*AQ','STAGE=SW+SS+2*AQ')
src=src.replace('int N,int K,int R,float* out,int ld,float alpha,float beta,WeightLayout layout,bool twice)','int N,int K,int R,float* out,int ld,float alpha,float beta,WeightLayout layout,Digits sparse)')
src=src.replace('ng*576+3*Geo::AQ','ng*576+2*Geo::AQ')
src=src.replace('for(int p=0;p<3;p++)','for(int p=0;p<2;p++)')
src=src.replace('unsigned b[3][4]','unsigned b[2][4]')
src=src.replace('float d[3][4]','float d[2][4]')
old='''          float factor=twice?2.f:1.f;
          int dot=__float2int_rn(factor*d[0][i])+4*__float2int_rn(factor*d[1][i])+16*__float2int_rn(factor*d[2][i]);'''
new='''          int t=t0+n*8+tq*2+(i&1),wrow=16*(m%2)+gid+(i>>1)*8;
          unsigned bits[4];
#pragma unroll
          for(int strip=0;strip<4;strip++)bits[strip]=*(const uint32_t*)(s+strip*Geo::NG*128+(gw+m/2)*128+wrow*4);
          size_t packet=(size_t(t/64)*all+kb0+kb)*64+t%64;
          unsigned count=t<R?sparse.counts[packet]:0;int correction=0,dot=0;
          if(count<=ListCapacity){
            for(int j=0;j<count;j++){
              unsigned e=sparse.entries[packet*ListCapacity+j];int pos=e&127,value=int(int8_t(e>>8));
              correction+=((bits[pos>>5]>>(pos&31))&1)?value:-value;
            }
            dot=__float2int_rn(2.f*d[0][i])+4*__float2int_rn(2.f*d[1][i])+16*correction;
          }else{
#pragma unroll 4
            for(int pos=0;pos<128;pos++){
              int value=int(carrier.q[(size_t(t)*all+kb0+kb)*128+pos]);
              dot+=((bits[pos>>5]>>(pos&31))&1)?value:-value;
            }
          }'''
assert old in src
src=src.replace(old,new)
src=src.replace('cudaStream_t st,Encoding encoding)','cudaStream_t st,Digits sparse)')
src=src.replace('p.layout,encoding==Encoding::ExactA8','p.layout,sparse')
src=src.replace('st,d.encoding','st,d')
src=src.replace('auto e=validate(p);','auto e=sparse_fp4::validate(p);').replace('size_t need=partial_floats(p);','size_t need=sparse_fp4::partial_floats(p);')
src=src.replace('!w||!s||!c.scales','!w||!s||!c.q||!c.scales')
src=src.replace('(d.encoding!=Encoding::Signed7&&d.encoding!=Encoding::ExactA8)||','(!d.counts||d.count_capacity<count_bytes(p.k,p.rows)||!d.entries||d.entry_capacity<entry_words(p.k,p.rows)||uintptr_t(d.entries)%2)||')
(E/'sparse_digits.cu').write_text(src)
proof='''import Std
namespace SparseA8

def labels : List Nat := ['''+','.join(map(str,vals))+''']
def label (i : Fin 256) : Nat := labels[i.val]!
def twiceE2M1 (code : Nat) : Int :=
  (if code ≥ 8 then -1 else 1) * Int.ofNat
    (match code % 8 with | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3 | 4 => 4 | 5 => 6 | 6 => 8 | _ => 12)
def third (i : Fin 256) : Int :=
  let v := (label i) / 256
  if v ≥ 128 then Int.ofNat v - 256 else Int.ofNat v
set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
theorem exact_all_byte_patterns : ∀ i : Fin 256,
  Int.ofNat i.val - 128 = twiceE2M1 (label i % 16) +
    4 * twiceE2M1 ((label i / 16) % 16) + 16 * third i := by decide
set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
theorem third_bound : ∀ i : Fin 256, (third i).natAbs ≤ 6 := by decide
end SparseA8
'''
(E/'sparse-digit-results'/'sparse_a8.lean').write_text(proof)
print('generated',len(src),'bytes; all 256 balanced codes exact')
