#pragma once
static uint32_t binary_e2m1(unsigned x){
 uint32_t word=0;for(unsigned i=0;i<8;i++)word|=(2u|(((x>>i)&1u)<<3))<<(4*i);return word;
}
static unsigned signs(uint32_t word,unsigned half){
 word>>=half*16;return ((word>>3)&1u)|((word>>6)&2u)|((word>>9)&4u)|((word>>12)&8u);
}
struct QFragment {
 uint32_t a[32][2][2][4];
 uint16_t scale[8][2][2];
 explicit QFragment(const uint8_t*st,unsigned g){
  for(unsigned lane=0;lane<32;lane++)for(unsigned mm=0;mm<2;mm++)for(unsigned h=0;h<2;h++){
   uint32_t pair;std::memcpy(&pair,st+g*512+lane*16+(mm*2+h)*4,4);
   a[lane][mm][h][0]=binary_e2m1(pair);a[lane][mm][h][2]=binary_e2m1(pair>>8);
   a[lane][mm][h][1]=binary_e2m1(pair>>16);a[lane][mm][h][3]=binary_e2m1(pair>>24);
  }
  for(unsigned gid=0;gid<8;gid++)for(unsigned mm=0;mm<2;mm++)for(unsigned r=0;r<2;r++)
   std::memcpy(&scale[gid][mm][r],st+SW+g*64+(gid*4+mm*2+r)*2,2);
 }
};
static int32_t literal_half(const QFragment&fragment,const LUT&lut,size_t base,unsigned mm,unsigned h,unsigned r,unsigned gid){
 int32_t dot=0;for(unsigned block=0;block<8;block++){
  unsigned tq=block%4,upper=block/4,lane=gid*4+tq;uint32_t word=fragment.a[lane][mm][h][r+upper*2];
  unsigned kb=h*8+block;
  dot+=lut.native[base+kb*2*16+signs(word,0)];dot+=lut.native[base+(kb*2+1)*16+signs(word,1)];
 }return dot;
}
static bool shortened(const std::array<Tape,2>&tapes,const std::array<LUT,2>&lut,const source14::Packed&p,unsigned B,bool cand,const std::array<std::vector<uint32_t>,3>&saved,const std::array<std::vector<uint64_t>,3>&traces,Counts&counts){
 unsigned ng=p.F/32,ncta=(ng+7)/8;std::vector<std::vector<Output>>dictionary(ncta);std::vector<unsigned>seen(size_t(B)*p.F);
 for(unsigned cta=0;cta<ncta;cta++)dictionary[cta]=cand?candidate(cta*8,ng,B,seen,counts.inactive):strongest(cta*8,ng,B,seen,counts.inactive);
 if(!std::all_of(seen.begin(),seen.end(),[](unsigned n){return n==1;}))return false;
 std::atomic<bool>good{true};
 #pragma omp parallel for num_threads(8) schedule(static)
 for(unsigned cta=0;cta<ncta;cta++){
  unsigned g0=cta*8,live=std::min(8u,ng-g0);const auto&dict=dictionary[cta];size_t extent=dict.size();
  std::vector<size_t>index(8*32*32,~size_t(0));for(size_t i=0;i<extent;i++){auto[f,row]=dict[i];index[((f/32-g0)*32+row)*32+f%32]=i;}
  std::array<std::vector<float>,2>acc{std::vector<float>(extent),std::vector<float>(extent)},rawacc=acc;
  std::array<std::vector<uint64_t>,2>digest{std::vector<uint64_t>(extent,14695981039346656037ull),std::vector<uint64_t>(extent,14695981039346656037ull)};
  std::array<std::vector<unsigned>,2>fold_owner{std::vector<unsigned>(extent),std::vector<unsigned>(extent)};
  std::array<std::array<uint8_t,STAGE>,8>ring{};std::array<std::array<float,32>,2>globals{};
  for(unsigned pass=0;pass<2;pass++)for(unsigned row=0;row<B;row++)globals[pass][row]=tapes[pass].g[row];
  for(unsigned kb=0;kb<KB;kb++){
   auto&st=ring[kb%8];std::fill(st.begin(),st.end(),0xcd);
   std::memcpy(st.data(),p.w.data()+(size_t(kb)*ng+g0)*512,live*512);
   std::memcpy(st.data()+SW,p.s.data()+(size_t(kb)*ng+g0)*32,live*64);
   for(unsigned pass=0;pass<2;pass++){
    unsigned qo=SW+SWS+pass*(SA+SAS),so=qo+SA;
    std::memcpy(st.data()+qo,tapes[pass].q.data()+size_t(kb)*64*64,SA);
    std::memcpy(st.data()+so,tapes[pass].s.data()+size_t(kb)*64*8,SAS);
    for(unsigned row=0;row<32;row++)for(unsigned k=0;k<128;k+=2)if(st[qo+row*64+(((k/32)^((row>>1)&3))*16)+k%32/2]!=tapes[pass].q[qoff(row,kb,k)])good=false;
    for(unsigned row=0;row<32;row++)for(unsigned k=0;k<8;k++)if(st[so+row*8+k]!=tapes[pass].s[(size_t(kb)*64+row)*8+k])good=false;
   }
   unsigned FW=cand?4:8,WG=cand?2:1;
   for(unsigned warp=0;warp<FW;warp++)for(unsigned q=0;q<WG;q++){
    unsigned g=warp*WG+q;QFragment fragment(st.data(),g);
    for(unsigned pass=0;pass<2;pass++)for(unsigned n=0;n<4;n++)for(unsigned mm=0;mm<2;mm++)
     for(unsigned gid=0;gid<8;gid++)for(unsigned tq=0;tq<4;tq++)for(unsigned r=0;r<2;r++)for(unsigned j=0;j<2;j++){
      unsigned row=n*8+tq*2+j,f=(g0+g)*32+mm*16+r*8+gid;
      if(g>=live||row>=B)continue;
      size_t i=index[(g*32+row)*32+f%32];if(i>=extent){good=false;continue;}
      if(fold_owner[pass][i]!=kb)good=false;
      ++fold_owner[pass][i];
      auto*raw=p.raw.data()+(size_t(f)*KB+kb)*18;size_t a=(size_t(row)*KB+kb)*16*256,b=(size_t(row)*KB+kb)*32*16;
      int32_t raw0=0,raw1=0;for(unsigned block=0;block<8;block++){raw0+=lut[pass].raw[a+block*256+raw[2+block]];raw1+=lut[pass].raw[a+(8+block)*256+raw[10+block]];}
      int32_t d=literal_half(fragment,lut[pass],b,mm,0,r,gid);
      if(d!=raw0)good=false;
      int32_t h1=literal_half(fragment,lut[pass],b,mm,1,r,gid);if(h1!=raw1)good=false;d+=h1;
      uint16_t ns=fragment.scale[gid][mm][r],rs=raw[0]|uint16_t(raw[1])<<8;if(ns!=rs)good=false;
      rawacc[pass][i]=std::fma(std::ldexp(float(raw0+raw1),-10),source14::full(rs),rawacc[pass][i]);
      acc[pass][i]=std::fma(std::ldexp(float(d),-10),source14::full(ns),acc[pass][i]);
      if(std::bit_cast<uint32_t>(acc[pass][i])!=std::bit_cast<uint32_t>(rawacc[pass][i]))good=false;
      feed(digest[pass][i],uint32_t(d));feed(digest[pass][i],ns);feed(digest[pass][i],std::bit_cast<uint32_t>(acc[pass][i]));
     }
   }
  }
  std::array<std::vector<unsigned>,2>owners{std::vector<unsigned>(extent,~0u),std::vector<unsigned>(extent,~0u)};
  for(size_t i=0;i<extent;i++){
   auto[f,row]=dict[i];float primary=add(mul(acc[0][i],globals[0][row]),0.f),final=add(mul(acc[1][i],globals[1][row]),primary);
   float rawprimary=add(mul(rawacc[0][i],tapes[0].g[row]),0.f),rawfinal=add(mul(rawacc[1][i],tapes[1].g[row]),rawprimary);
   if(std::bit_cast<uint32_t>(primary)!=std::bit_cast<uint32_t>(rawprimary)||std::bit_cast<uint32_t>(final)!=std::bit_cast<uint32_t>(rawfinal))good=false;
   for(unsigned pass=0;pass<2;pass++){
    if(fold_owner[pass][i]!=KB||owners[pass][i]!=~0u)good=false;
    owners[pass][i]=0;
    size_t at=(size_t(pass)*B+row)*p.F+f;uint32_t word=std::bit_cast<uint32_t>(pass?final:primary);
    for(unsigned arm=0;arm<3;arm++)if(word!=saved[arm][at]||digest[pass][i]!=traces[arm][at])good=false;
   }
  }
  for(auto&pass:owners)if(!std::all_of(pass.begin(),pass.end(),[](unsigned status){return status==0;}))good=false;
 }
 counts.stage_q=uint64_t(ncta)*KB*2*SA;counts.stage_s=uint64_t(ncta)*KB*2*SAS;counts.stage_w=uint64_t(ng)*KB*512;counts.stage_ws=uint64_t(ng)*KB*64;
 counts.folds=uint64_t(2)*B*p.F*KB;counts.outputs=uint64_t(2)*B*p.F;counts.owners=counts.outputs;counts.global_scale_words=uint64_t(ncta)*2*B;counts.half_dots=counts.folds*2;
 return good.load();
}
