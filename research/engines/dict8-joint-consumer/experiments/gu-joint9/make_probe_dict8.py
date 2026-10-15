from pathlib import Path
import json,hashlib
p=Path(__file__).parent
s=(p/'probe.cu').read_text();s=s[:s.index('\n__global__ __launch_bounds__(288) void joint9_probe')]
s=s.replace('WF == 8 || WF == 9) ? 832','WF == 8 || WF == 9 || WF == 10) ? 832')
s=s.replace('CS = WF == 1 ? 256 : (WF >= 4 && WF != 8) ? 128 : 64','CS = WF == 1 ? 256 : (WF >= 4 && WF != 8 && WF != 10) ? 128 : 64')
s=s.replace('WF == 3 || WF == 9,','WF == 3 || WF == 9 || WF == 10,')
# Frozen serving predates the H1 global-book argument. The compiler-only
# successor owns its explicit dictionary argument; native0 explicitly has none.
old='const TailArgs& ta = TailArgs{}) {'
assert s.count(old)==1
s=s.replace(old,'const TailArgs& ta, const uint32_t* global_book) {')
a='''    if constexpr (WF == 9) {
      static_assert(WG==2 && MODE==3 && CARRIER);
      uint32_t labels[13],g[26],u[26];'''
b='''    if constexpr (WF == 10) {
      static_assert(WG==2 && MODE==3 && CARRIER);
      // Each original native word is eight trits -> a 13-bit base3 code.
      // Gate/up sibling codes are adjacent; 32 codes occupy exactly 52B/lane.
      uint32_t labels[13];
#pragma unroll
      for(int i=0;i<13;i++)labels[i]=*(const uint32_t*)(st+gw*832+i*128+lane*4);
#pragma unroll
      for(int w=0;w<16;w++) {
        constexpr unsigned mask=8191;
        int b=(26*w)/32,sh=(26*w)%32;
        unsigned v=__funnelshift_r(labels[b],b+1<13?labels[b+1]:0,sh);
        a[w/8][(w/4)%2][w%4]=global_book[v&mask];
        a[2+w/8][(w/4)%2][w%4]=global_book[(v>>13)&mask];
      }
    } else if constexpr (WF == 9) {
      static_assert(WG==2 && MODE==3 && CARRIER);
      uint32_t labels[13],g[26],u[26];'''
assert s.count(a)==1;s=s.replace(a,b)
s+='''
__global__ __launch_bounds__(288) void dict8_probe(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means,const uint32_t*dict){
 gemm_fp4_body<8,2,32,3,10,0,false,true>(A,AS,AG,R,K,W,nullptr,N,out,ldo,means,FuseArgs{},TailArgs{},dict);
}
__global__ __launch_bounds__(288) void native0_probe(const uint8_t*A,const uint8_t*AS,const float*AG,int R,int K,const uint8_t*W,int N,float*out,int ldo,const float*means){
 gemm_fp4_body<8,2,32,3,0,0,false,true>(A,AS,AG,R,K,W,nullptr,N,out,ldo,means,FuseArgs{},TailArgs{},nullptr);
}
} // h5
'''
assert 'WF == 10) ? 832' in s and 'WF != 10' in s
(p/'probe-dict8.cu').write_text(s)
r={'parent_probe_sha256':hashlib.sha256((p/'probe.cu').read_bytes()).hexdigest(),'generated_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'dictionary_entries':6561,'dictionary_bytes':26244,'prepared_weight_payload_ratio_vs_WF0':0.8125,'scope':'Original separate operand words and per-output native contraction/epilogue schedule; no RN/RTZ distribution assumption. Compiler probe only.'}
(p/'dict8-source.json').write_text(json.dumps(r,indent=2));print(r)
# Derive a new guarded CPU compiler generation, never replay v1.
script=(p/'make_remote_probe.py').read_text().replace("p/'probe.cu'","p/'probe-dict8.cu'").replace('gu-joint9-compile-v1','gu-joint9-dict8-compile-v2')
(p/'make_remote_dict8.py').write_text(script)
