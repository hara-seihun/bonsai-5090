from pathlib import Path
import hashlib,json
p=Path(__file__).parent
original=Path('~/work/five-wide-native/gu-scale-absorb-cpu-v1/fp4.cu').read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='0754bd730848620791da50c9aaadd6ba0a8f431434a75428b916969464164f52'
marker='\ntemplate <int FW, int WG, int TOKW, int MODE, int WF, int PIPE = 0, bool MICRO = false, bool CARRIER = false, bool DEGREE8 = false>\n__global__'
s=original[:original.index(marker)]
def change(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
change('(WF == 3 || WF == 8) ? 832','(WF == 3 || WF == 8 || WF == 9) ? 832')
change('((WF == 3 || WF == 8) ? 1024 : 0)','(WF == 9 ? 2048 : (WF == 3 || WF == 8) ? 1024 : 0)')
change('static_assert(!CARRIER || WF == 0 || WF == 3,','static_assert(!CARRIER || WF == 0 || WF == 3 || WF == 9,')
start=s.index('  if (warp == FW) {')
setup=s[s.index('  uint32_t* book = nullptr;'):start]
new_setup='''  uint32_t* book = nullptr;
  if constexpr (WF == 9) {
    book = (uint32_t*)(smem + G::SMEM - 2048);
    for (int code=tid;code<512;code+=blockDim.x) {
      unsigned c=code&255,g=0,u=0,ng=0,nu=0;
      for (int j=0;j<5;j++,c/=3) {
        unsigned d=c%3,v=d==0?10u:d==2?2u:0u;
        if (((j+(code>>8))&1)==0) {g|=v<<(4*ng++);}else{u|=v<<(4*nu++);}
      }
      book[code]=g|(u<<16);
    }
  } else if constexpr (WF == 3 || WF == 8) {
    book = (uint32_t*)(smem + G::SMEM - 1024);
    for(int code=tid;code<256;code+=blockDim.x) {
      unsigned q=code, value=0;
#pragma unroll
      for(int j=0;j<5;j++,q/=3) value|=(q%3==0?10u:q%3==2?2u:0u)<<(4*j);
      book[code]=value;
    }
  }
  __syncthreads();
  trace.end(micro::Setup);

'''
assert setup in s;s=s.replace(setup,new_setup,1)
opening='''#pragma unroll
    for (int q = 0; q < WG; q++) {
      if constexpr (WF == 4 || WF == 5) {'''
replacement='''    if constexpr (WF == 9) {
      static_assert(WG==2 && MODE==3 && CARRIER);
      uint32_t labels[13],g[26],u[26];
#pragma unroll
      for(int i=0;i<13;i++)labels[i]=*(const uint32_t*)(st+gw*832+i*128+lane*4);
#pragma unroll
      for(int i=0;i<26;i++) {
        unsigned word=labels[i/2],a0=book[(word>>(16*(i%2)))&255],a1=book[256+((word>>(16*(i%2)+8))&255)];
        g[i]=(a0&0xfff)|((a1&255)<<12);u[i]=(a0>>16)|((a1>>16)<<8);
      }
#pragma unroll
      for(int w=0;w<16;w++) {
        constexpr unsigned mask=0xffffffffu;
        int b=(8*w)/5,sh=((8*w)%5)*4;
        unsigned gv=(g[b]>>sh)|(g[b+1]<<(20-sh)),uv=(u[b]>>sh)|(u[b+1]<<(20-sh));
        if(sh>=12){gv|=g[b+2]<<(40-sh);uv|=u[b+2]<<(40-sh);}
        a[w/8][(w/4)%2][w%4]=gv&mask;
        a[2+w/8][(w/4)%2][w%4]=uv&mask;
      }
    } else {
#pragma unroll
    for (int q = 0; q < WG; q++) {
      if constexpr (WF == 4 || WF == 5) {'''
change(opening,replacement)
closing='''    trace.end(micro::Decode, kb);'''
change(closing,'    } // original independent-weight decoder\n'+closing)
s+='''
__global__ __launch_bounds__(288) void joint9_probe(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means){
 gemm_fp4_body<8,2,32,3,9,0,false,true>(A,AS,AG,R,K,W,nullptr,N,out,ldo,means,FuseArgs{});
}
__global__ __launch_bounds__(288) void native0_probe(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means){
 gemm_fp4_body<8,2,32,3,0,0,false,true>(A,AS,AG,R,K,W,nullptr,N,out,ldo,means,FuseArgs{});
}
__global__ __launch_bounds__(288) void independent3_probe(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means){
 gemm_fp4_body<8,2,32,3,3,0,false,true>(A,AS,AG,R,K,W,nullptr,N,out,ldo,means,FuseArgs{});
}
} // h5
'''
(p/'probe.cu').write_text(s)
(p/'probe-source.json').write_text(json.dumps({'frozen_source_sha256':hashlib.sha256(original.encode()).hexdigest(),'generated_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'scope':'Matched complete K loop, unchanged native MMA call sequence and terminal GU observer; generated source is a compiler probe, not a linked/accepted model route.'},indent=2))
print('generated',len(s),'bytes')
