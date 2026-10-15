#!/usr/bin/env python3
from pathlib import Path
E=Path(__file__).resolve().parent
s=(E/'signed_digits_bench.cu').read_text()
s=s.replace('#include "signed_digits.cuh"','#include "sparse_digits.cuh"\n#include <fstream>\n#include <cstring>')
s=s.replace('signed_fp4::Digits d{m.alloc<uint8_t>(signed_fp4::digit_bytes(k,r)),signed_fp4::digit_bytes(k,r),signed_fp4::Encoding::ExactA8};','sparse_fp4::Digits d{m.alloc<uint8_t>(sparse_fp4::digit_bytes(k,r)),sparse_fp4::digit_bytes(k,r),m.alloc<uint8_t>(sparse_fp4::count_bytes(k,r)),sparse_fp4::count_bytes(k,r),m.alloc<uint16_t>(sparse_fp4::entry_words(k,r)),sparse_fp4::entry_words(k,r)};')
s=s.replace('signed_fp4::encode(c,k,r,d,st)','sparse_fp4::encode(c,k,r,d,st)')
s=s.replace('signed_fp4::project(p,weights,scales,c,d','sparse_fp4::project(p,weights,scales,c,d')
s=s.replace('signed_fp4::project(p,signs,ws,c,d','sparse_fp4::project(p,signs,ws,c,d')
s=s.replace('CUDA(signed_fp4::configure(signed_fp4::Schedule(cfg)));','CUDA(signed_fp4::configure(signed_fp4::Schedule(cfg)));CUDA(sparse_fp4::configure(sparse_fp4::Schedule(cfg)));')
s=s.replace('if(argc!=5)','if(argc!=6)')
s=s.replace('signed_digits_bench MODEL witness|gu|down|head R OUTPUT','sparse_digits_bench MODEL witness|down R OUTPUT CAPTURED_H_F32')
s=s.replace('(!wit&&kind!="down"&&kind!="gu"&&kind!="head")','(!wit&&kind!="down")')
s=s.replace('for(size_t i=0;i<hx.size();i++)hx[i]=float(int(mix(i+999)%65536)-32768)/32761;','''std::ifstream input(argv[5],std::ios::binary);if(!input.read(reinterpret_cast<char*>(hx.data()),hx.size()*4)){fprintf(stderr,"incomplete captured layer0 hidden rows\\n");return 2;}''')
s=s.replace('a8_fp4::Digits d4{','signed_fp4::Digits d3{m.alloc<uint8_t>(signed_fp4::digit_bytes(k,r)),signed_fp4::digit_bytes(k,r),signed_fp4::Encoding::ExactA8};\n  a8_fp4::Digits d4{')
s=s.replace('CUDA(sparse_fp4::encode(c,k,r,d,st));CUDA(a8_fp4::encode(c,k,r,d4,st));','CUDA(sparse_fp4::encode(c,k,r,d,st));CUDA(signed_fp4::encode(c,k,r,d3,st));CUDA(a8_fp4::encode(c,k,r,d4,st));')
s=s.replace('if(cfg>=0&&!full8){a8_fp4::Plan p4{n,k,r,split,a8_fp4::Schedule(cfg),a8_fp4::WeightLayout::Packed1};CUDA(a8_fp4::encode(c,k,r,d4,st));CUDA(a8_fp4::project(p4,signs,ws,c,d4,control,n,partial,out*16,1,0,st));','if(cfg>=0&&!full8){signed_fp4::Plan p3{n,k,r,split,signed_fp4::Schedule(cfg),signed_fp4::WeightLayout::Packed1};CUDA(signed_fp4::encode(c,k,r,d3,st));CUDA(signed_fp4::project(p3,signs,ws,c,d3,control,n,partial,out*16,1,0,st));')
s=s.replace('if(hc[i]!=hy[i])','if(memcmp(&hc[i],&hy[i],4)!=0)')
s=s.replace('same-plan 3/4 mismatch','same-plan sparse/exact3 mismatch').replace('same_plan_four_digit_bitexact','same_plan_exact3_bitexact')
s=s.replace('record("exact3_given8",name,cfg,split,false,cont,[&](){if(incl)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));if(boundary>=1)CUDA(sparse_fp4::encode(c,k,r,d,st));CUDA(sparse_fp4::project(p,signs,ws,c,d,y,n,partial,out*16,1,0,st));if(cont)continuation();});','''record("sparse2_given8",name,cfg,split,false,cont,[&](){if(incl)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));if(boundary>=1)CUDA(sparse_fp4::encode(c,k,r,d,st));CUDA(sparse_fp4::project(p,signs,ws,c,d,y,n,partial,out*16,1,0,st));if(cont)continuation();});
      record("exact3_given8",name,cfg,split,false,cont,[&](){if(incl)CUDA(bitserial::produce(bitserial::Route::Dp4a01,x,k,r,c,st));if(boundary>=1)CUDA(signed_fp4::encode(c,k,r,d3,st));CUDA(signed_fp4::project(p,signs,ws,c,d3,y,n,partial,out*16,1,0,st));if(cont)continuation();});''')
s=s.replace('complete exact3','complete sparse2')
s=s.replace('if(signed_fp4::encode(c,k,r,{d.data,d.capacity-1,d.encoding},st)!=cudaErrorInvalidValue)exit(3);','''auto short_d=d;short_d.capacity--;if(sparse_fp4::encode(c,k,r,short_d,st)!=cudaErrorInvalidValue)exit(3);
    short_d=d;short_d.count_capacity--;if(sparse_fp4::encode(c,k,r,short_d,st)!=cudaErrorInvalidValue)exit(3);
    short_d=d;short_d.entry_capacity--;if(sparse_fp4::encode(c,k,r,short_d,st)!=cudaErrorInvalidValue)exit(3);''')
# Broaden native raw witness without changing producer/scaling tests.
s=s.replace('for(int r:{5,16,32,65}){','for(int mode=0;mode<4;mode++)for(int r:{5,16,32,65}){')
s=s.replace('q[i]=int8_t(i%256-128);','q[i]=mode==0?int8_t(i%256-128):mode==1?(i%16==0?int8_t(127):int8_t(int(i%7)-3)):mode==2?int8_t(127):((i%128==(i/128)%128)?int8_t((i/128)%256-128):int8_t(0));')
s=s.replace('size_t count=size_t(r+2)*ld;','signed_fp4::Digits wd3{m.alloc<uint8_t>(signed_fp4::digit_bytes(k,r)),signed_fp4::digit_bytes(k,r),signed_fp4::Encoding::ExactA8};CUDA(signed_fp4::encode(c,k,r,wd3,st));\n    size_t count=size_t(r+2)*ld;')
s=s.replace('auto* initial=m.alloc<float>(count);','auto* initial=m.alloc<float>(count);auto* control=m.alloc<float>(count);')
s=s.replace('check(active,expected,&max,&rel);','check(active,expected,&max,&rel);\n      CUDA(cudaMemcpyAsync(control,initial,count*4,cudaMemcpyDeviceToDevice,st));CUDA(signed_fp4::project(p,weights,scales,c,wd3,control+prefix,ld,partial,size_t(r)*n*8,.75f,.25f,st));CUDA(cudaStreamSynchronize(st));std::vector<float> dense(count);CUDA(cudaMemcpy(dense.data(),control,count*4,cudaMemcpyDeviceToHost));if(memcmp(dense.data(),hy.data(),count*4)){fprintf(stderr,"witness same-plan dense mismatch mode%d r%d cfg%d split%d\\n",mode,r,cfg,split);exit(3);}')
s=s.replace('\\\"kind\\\":\\\"witness\\\",\\\"layout\\\":\\\"%s\\\",','\\\"kind\\\":\\\"witness\\\",\\\"mode\\\":%d,\\\"same_plan_exact3_bitexact\\\":true,\\\"layout\\\":\\\"%s\\\",')
s=s.replace('layout==signed_fp4::WeightLayout::WF8?"wf8":"packed1",r,cfg,split,max,rel);','mode,layout==signed_fp4::WeightLayout::WF8?"wf8":"packed1",r,cfg,split,max,rel);')
s=s.replace('auto invalid=p;invalid.layout=signed_fp4::WeightLayout(99);','auto invalid=p;invalid.layout=signed_fp4::WeightLayout::WF8;if(sparse_fp4::validate(invalid)!=cudaErrorInvalidValue)exit(3);invalid.layout=signed_fp4::WeightLayout(99);')
s=s.replace('"kind\\\":\\\"witness\\\",\\\"layout', '"kind\\\":\\\"witness\\\",\\\"layout') if False else s
# Print native list counts and assert source q equals original float32 packet formula.
needle='size_t trashn=128*1024*1024/4;'
insert='''std::vector<uint8_t> counts(sparse_fp4::count_bytes(k,r));CUDA(cudaMemcpy(counts.data(),d.counts,counts.size(),cudaMemcpyDeviceToHost));
  std::vector<float> sx(size_t(r)*k/128);CUDA(cudaMemcpy(sx.data(),c.scales,sx.size()*4,cudaMemcpyDeviceToHost));
  size_t nz=0,fallback=0;unsigned max_count=0;
  for(int t=0;t<r;t++)for(int g=0;g<k/128;g++){
    unsigned count=counts[(size_t(t/64)*(k/128)+g)*64+t%64];nz+=count;fallback+=count>sparse_fp4::ListCapacity;max_count=std::max(max_count,count);
    float maxabs=0;for(int j=0;j<128;j++)maxabs=std::max(maxabs,std::abs(hx[size_t(t)*k+g*128+j]));
    float inverse=maxabs==0?0:127.f/maxabs;float expected_scale=maxabs/127.f;
    if(memcmp(&expected_scale,&sx[size_t(t)*(k/128)+g],4)){fprintf(stderr,"sx source mismatch\\n");return 3;}
    for(int j=0;j<128;j++)if(q[size_t(t)*k+g*128+j]!=int(std::nearbyint(hx[size_t(t)*k+g*128+j]*inverse))){fprintf(stderr,"q source mismatch\\n");return 3;}
  }
  fprintf(file,"{\\"kind\\":\\"captured_packet\\",\\"r\\":%d,\\"k\\":%d,\\"nz_fraction\\":%.9g,\\"dense_fallback_groups\\":%zu,\\"max_third_count\\":%u,\\"list_capacity\\":%d,\\"original_q_sx_bitexact\\":true}\\n",r,k,double(nz)/(r*k),fallback,max_count,sparse_fp4::ListCapacity);fflush(file);
  '''
assert needle in s
s=s.replace(needle,insert+needle)
(E/'sparse_digits_bench.cu').write_text(s)
print('generated bench',len(s))
