from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]; out=Path(__file__).resolve().parent
src=(root/'src/dict8.cu').read_text()
assert hashlib.sha256(src.encode()).hexdigest()=='043509429b79c58c3a4647e23b6b1738c7b7a9d025f2f889b65577d1708f86d3'
(out/'baseline.cu').write_text(src)
beg=src.index('      // Each original native word')
end=src.index('    } else if constexpr (WF == 9)',beg)
for route,shared in [('split53_shared',1152),('split53_shuffle',1024)]:
 s=src.replace('(WF == 9 ? 2048 : (WF == 3 || WF == 8) ? 1024 : 0)',f'(WF == 10 ? {shared} : WF == 9 ? 2048 : (WF == 3 || WF == 8) ? 1024 : 0)')
 init='''  unsigned book3_lane=0;
  if constexpr (WF == 10) {
    book = (uint32_t*)(smem + G::SMEM - SPLIT_BYTES);
    if(tid<243) {
      unsigned q=tid,value=0;
#pragma unroll
      for(int j=0;j<5;j++,q/=3) value|=(q%3==0?10u:q%3==2?2u:0u)<<(4*j);
      book[tid]=value;
    }
    unsigned q=lane;
#pragma unroll
    for(int j=0;j<3;j++,q/=3)book3_lane|=(q%3==0?10u:q%3==2?2u:0u)<<(4*j);
    THREE_STORE
  }
'''.replace('SPLIT_BYTES',str(shared)).replace('THREE_STORE','if(tid<27)book[256+tid]=book3_lane;' if route.endswith('shared') else '')
 s=s.replace('  uint32_t* book = nullptr;','  uint32_t* book = nullptr;\n'+init)
 decoder='''      // Split coordinates: all low five-trit labels first, then high three-trit labels.
      // Exactly 8*32 + 5*32 bits/lane. No array of expanded g/u intermediates.
      uint32_t labels[13];
#pragma unroll
      for(int i=0;i<13;i++)labels[i]=*(const uint32_t*)(st+gw*832+i*128+lane*4);
#pragma unroll
      for(int w=0;w<16;w++) {
#pragma unroll
        for(int arm=0;arm<2;arm++) {
          int i=2*w+arm, b=8+(5*i)/32, sh=(5*i)%32;
          unsigned hi;
          if(sh>27) hi=__funnelshift_r(labels[b],labels[b+1],sh)&31;
          else hi=(labels[b]>>sh)&31;
          unsigned lo=(labels[i/4]>>(8*(i%4)))&255;
          unsigned upper=THREE_READ;
          a[arm*2+w/8][(w/4)%2][w%4]=book[lo]|(upper<<20);
        }
      }
'''.replace('THREE_READ','book[256+hi]' if route.endswith('shared') else '__shfl_sync(0xffffffff,book3_lane,hi)')
 # index positions changed by insertions; replace original block by exact matching slice.
 s=s.replace(src[beg:end],decoder)
 s=s.replace('||!dict||R!=64','||R!=64')
 (out/(route+'.cu')).write_text(s)
s=src.replace('(WF == 9 ? 2048 : (WF == 3 || WF == 8) ? 1024 : 0)','(WF == 10 ? 26368 : WF == 9 ? 2048 : (WF == 3 || WF == 8) ? 1024 : 0)')
s=s.replace('  uint32_t* book = nullptr;','  uint32_t* book = nullptr;\n  if constexpr(WF==10) {\n    book=(uint32_t*)(smem+G::SMEM-26368);\n    for(int code=tid;code<6561;code+=blockDim.x)book[code]=global_book[code];\n  }')
s=s.replace('=global_book[v&mask]','=book[v&mask]').replace('=global_book[(v>>13)&mask]','=book[(v>>13)&mask]')
(out/'shared8.cu').write_text(s)
s=s.replace('SWS = NG * WFmt<WF>::CS','SWS = WF == 10 ? 0 : NG * WFmt<WF>::CS')
(out/'shared8_compact.cu').write_text(s)
s=s.replace('gemm_fp4_body<8,2,32,3,WF,4,','gemm_fp4_body<4,2,32,3,WF,2,').replace('WsGeom<8,2,32,WF,4>','WsGeom<4,2,32,WF,2>').replace('dim3(2,68,1),dim3(288)','dim3(2,136,1),dim3(160)').replace('__launch_bounds__(288)','__launch_bounds__(160)')
(out/'shared8_dual.cu').write_text(s)
(out/'native0.cu').write_text(src.replace('dispatch<10,false>','dispatch<0,false>'))
# Ten 24-trit groups plus two raw native-word halves per lane: 54B/lane.
literal='''      const uint8_t* packed=st+gw*864;
#pragma unroll
      for(int group=0;group<10;group++){
        unsigned L=*(const unsigned*)(packed+group*128+lane*4);
        unsigned H=packed[1280+group*32+lane];
        unsigned out3[3];lc_decode(L,H,out3);
#pragma unroll
        for(int j=0;j<3;j++){
          int q=3*group+j,w=q/2,arm=q%2;
          a[arm*2+w/8][(w/4)%2][w%4]=out3[j];
        }
      }
      unsigned raw=*(const unsigned*)(packed+1600+lane*4);
      a[1][1][3]=raw&0xaaaaaaaa;
      a[3][1][3]=(raw<<1)&0xaaaaaaaa;
'''
s=src.replace(src[beg:end],literal).replace('#include "fp4_ops.cuh"','#include "fp4_ops.cuh"\n#include "literal.cuh"')
s=s.replace('CW = WF == 1','CW = WF == 10 ? 864 : WF == 1').replace('SWS = NG * WFmt<WF>::CS','SWS = WF == 10 ? 0 : NG * WFmt<WF>::CS').replace('||!dict||R!=64','||R!=64')
(out/'literal3.cu').write_text(s)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.cu')}
(out/'generated.json').write_text(json.dumps({'base_commit':'3fdbad4547e51455b82d5e7e3f5d6194081449cb','sources':manifest,'scope':'CPU compilation only; original arithmetic/observer untouched'},indent=2)+'\n')
