#pragma once
#include <fstream>
#include <cstdint>
#include <map>
#include <vector>
#include <string>
#include <variant>
#include <cmath>
#include <cstring>
namespace source14 {
enum class Error {io,tensor,shape,nonfinite};
template<class T>using Result=std::variant<T,Error>;
struct Weights {unsigned K,F;std::vector<uint8_t>raw;};
struct Tensor {unsigned type;uint64_t offset,bytes;unsigned K,F;};
inline float full(uint16_t h){uint32_t sign=uint32_t(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,u;if(e==0){if(!m)u=sign;else{int exponent=-14;while(!(m&1024)){m<<=1;--exponent;}u=sign|uint32_t(exponent+127)<<23|(m&1023)<<13;}}else if(e==31)u=sign|0x7f800000u|m<<13;else u=sign|(e+112)<<23|m<<13;float f;std::memcpy(&f,&u,4);return f;}
class Model {
 std::ifstream file_;std::map<std::string,Tensor>tensors_;
 public:
 bool open(const std::string&index){std::ifstream in(index);std::string name;Tensor t;while(in>>name>>t.type>>t.offset>>t.bytes>>t.K>>t.F){if(!t.K||!t.F||!t.bytes||!tensors_.emplace(name,t).second)return false;}file_.open("~/data/bonsai1/Bonsai-27B-Q1_0.gguf",std::ios::binary);return in.eof()&&file_.is_open()&&tensors_.size()==851;}
 Result<std::vector<uint8_t>>bytes(const std::string&name,unsigned type,unsigned K,unsigned F){auto i=tensors_.find(name);if(i==tensors_.end()||i->second.type!=type||i->second.K!=K||i->second.F!=F)return Error::tensor;auto&t=i->second;std::vector<uint8_t>v(t.bytes);file_.clear();file_.seekg(t.offset);file_.read((char*)v.data(),v.size());if(!file_)return Error::io;return v;}
 Result<std::vector<float>>floats(const std::string&name,unsigned K,unsigned F=1){auto b=bytes(name,0,K,F);if(auto e=std::get_if<Error>(&b))return *e;auto&raw=std::get<std::vector<uint8_t>>(b);if(raw.size()!=uint64_t(K)*F*4)return Error::shape;std::vector<float>v(raw.size()/4);std::memcpy(v.data(),raw.data(),raw.size());for(auto x:v)if(!std::isfinite(x))return Error::nonfinite;return v;}
 Result<Weights>weights(std::initializer_list<std::pair<std::string,unsigned>>names,unsigned K){if(!K||K%128)return Error::shape;Weights w{K,0,{}};for(auto&[name,F]:names){auto b=bytes(name,41,K,F);if(auto e=std::get_if<Error>(&b))return *e;auto&v=std::get<std::vector<uint8_t>>(b);if(v.size()!=uint64_t(K/128)*F*18)return Error::shape;w.F+=F;w.raw.insert(w.raw.end(),v.begin(),v.end());}return w;}
 Result<std::vector<float>>embedding(unsigned token){auto i=tensors_.find("token_embd.weight");if(i==tensors_.end()||i->second.type!=41||i->second.K!=5120||token>=i->second.F)return Error::tensor;std::vector<uint8_t>raw(40*18);file_.clear();file_.seekg(i->second.offset+uint64_t(token)*raw.size());file_.read((char*)raw.data(),raw.size());if(!file_)return Error::io;std::vector<float>x(5120);for(unsigned g=0;g<40;g++){auto*p=raw.data()+18*g;float d=full(uint16_t(p[0])|uint16_t(p[1])<<8);for(unsigned k=0;k<128;k++)x[g*128+k]=(p[2+k/8]>>(k%8))&1?d:-d;}for(auto v:x)if(!std::isfinite(v))return Error::nonfinite;return x;}
};
}
