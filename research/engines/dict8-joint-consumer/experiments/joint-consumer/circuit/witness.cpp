#include "carrier.hpp"
#include <cstdio>
#include <cstdlib>
using namespace ternary5;
static void require(bool b,const char* m){if(!b){std::fprintf(stderr,"FAIL: %s\n",m);std::exit(1);}}
int main(){
    unsigned values[3]={0,2,10}; unsigned unique=0,res=0;
    std::array<unsigned,4096> counts{};
    for(unsigned x=0;x<32;x++){
        auto t=decode_raw(x);
        require(native(t[0])&&native(t[1])&&native(t[2]),"raw output validity");
        unsigned key=t[0]|(t[1]<<4)|(t[2]<<8);counts[key]++;
        auto r=Code5::parse(x);
        require(std::holds_alternative<Error>(r)==reserved(x),"host reserved status");
        res+=reserved(x);
    }
    std::puts("t0,t1,t2,code5_decimal,code5_hex");
    for(unsigned a:values)for(unsigned b:values)for(unsigned c:values){
        Triple t={std::uint8_t(a),std::uint8_t(b),std::uint8_t(c)};
        auto r=encode(t);require(std::holds_alternative<Code5>(r),"encode success");
        auto x=std::get<Code5>(r);require(decode(x)==t,"scalar roundtrip");
        unsigned key=a|(b<<4)|(c<<8);require(counts[key]>0,"surjection");unique++;
        std::printf("%X,%X,%X,%u,%02X\n",a,b,c,x.value(),x.value());
        for(unsigned lane=0;lane<8;lane++){
            std::array<std::uint8_t,24> src{};
            src[lane]=a;src[8+lane]=b;src[16+lane]=c;
            auto p=encode24(src);require(std::holds_alternative<Payload8>(p),"encode24");
            auto out=decode8(std::get<Payload8>(p));require(std::holds_alternative<std::array<U,3>>(out),"decode8");
            auto words=std::get<std::array<U,3>>(out);
            for(unsigned j=0;j<24;j++)require(((words[j/8]>>(4*(j%8)))&15)==src[j],"physical eight-set roundtrip");
        }
    }
    for(unsigned h=0;h<256;h++){
        U expect=0;for(unsigned j=0;j<8;j++)expect|=((h>>j)&1)<<(4*j);
        require(spread8(h)==expect,"all spread-byte inputs");
    }
    // Every pair of codes in every pair of slots, including noncanonical aliases.
    // This exercises all neighboring pollution patterns in the unmasked word route.
    unsigned physical=0;
    for(unsigned i=0;i<8;i++)for(unsigned j=i+1;j<8;j++)for(unsigned x=0;x<32;x++)for(unsigned y=0;y<32;y++){
        U l=0;unsigned h=0;std::array<unsigned,8> codes;codes.fill(24);codes[i]=x;codes[j]=y;
        for(unsigned k=0;k<8;k++){l|=(codes[k]&15)<<(4*k);h|=(codes[k]>>4)<<k;}
        auto words=decode8_prevalidated({l,std::uint8_t(h)});
        for(unsigned k=0;k<8;k++){
            auto t=decode_raw(codes[k]);for(unsigned z=0;z<3;z++)require(((words[z]>>(4*k))&15)==t[z],"raw physical locality");
        }
        physical++;
    }
    unsigned tail_cases=0;
    for(unsigned r=0;r<65536;r++){
        bool valid=true;U expected=0;
        for(unsigned j=0;j<8;j++){
            unsigned v=(r>>(2*j))&3;
            if(v==2)valid=false;
            expected|=(((v&1)<<1)|((v&2)<<2))<<(4*j);
        }
        auto out=decode_tail8(r);
        require(std::holds_alternative<U>(out)==valid,"compact tail validation");
        if(valid){require(std::get<U>(out)==expected,"compact tail exactness");tail_cases++;}
    }
    unsigned pair_cases=0;
    for(unsigned j=0;j<8;j++)for(unsigned g:values)for(unsigned u:values){
        U gate=g<<(4*j),up=u<<(4*j);
        auto r=encode_raw_pair(gate,up);require(std::holds_alternative<U>(r),"raw pair encode");
        auto d=decode_raw_pair(std::get<U>(r));require(std::holds_alternative<std::array<U,2>>(d),"raw pair decode");
        require(std::get<std::array<U,2>>(d)==std::array<U,2>{gate,up},"raw pair exactness");pair_cases++;
    }
    for(unsigned j=0;j<8;j++)for(unsigned r=0;r<16;r++){
        bool valid=native(r&10)&&native((r<<1)&10);
        require(std::holds_alternative<std::array<U,2>>(decode_raw_pair(r<<(4*j)))==valid,"raw pair reserved validation");
    }
    std::fprintf(stderr,"PASS: %u compact-tail states; %u raw-pair slot cases; 128 raw-pair validity cases.\n",tail_cases,pair_cases);
    auto bad=Code5::parse(32);require(std::get<Error>(bad)==Error::code_out_of_range,"out-of-range code");
    for(unsigned x=0;x<16;x++)if(!native(x))require(std::get<Error>(encode({std::uint8_t(x),0,0}))==Error::invalid_nibble,"invalid source nibble");
    require(unique==27 && res==5,"finite endpoint counts");
    std::fprintf(stderr,"PASS: 32 raw codes -> 27 native triples; 27 canonical roundtrips; 5 reserved host errors; 216 single-slot roundtrips; 256 spreads; %u exhaustive two-slot patterns.\n",physical);
}
