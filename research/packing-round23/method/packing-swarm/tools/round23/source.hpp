#pragma once
#include "../round18/grammar.hpp"
#include <memory>
#include <span>
#include <cstring>
#include <cmath>
namespace source19 {
enum class Region {layer_input,post_attention};
enum class Error {owner,shape,source,norm,manifest,nonfinite,duplicate_row,allocation};
template<class T>using Result=std::variant<T,Error>;
struct Identity {unsigned layer,batch,token,stream;uint64_t generation,boundary;Region region;bool operator==(const Identity&)const=default;};
class Image {
 Image(Identity id,readers18::Family f,std::string tensor,std::vector<float>sx,std::vector<float>sn,uint64_t sh,uint64_t nh,uint64_t dh):identity(id),family(std::move(f)),norm_tensor(std::move(tensor)),x(std::move(sx)),norm(std::move(sn)),source_hash(sh),norm_hash(nh),domain_hash(dh){}
 friend Result<std::shared_ptr<const Image>> supply(Identity,std::span<const float>,std::span<const float>,const std::string&,const readers18::Family&);
 public:
 const Identity identity;const readers18::Family family;const std::string norm_tensor;
 const std::vector<float> x,norm;const uint64_t source_hash,norm_hash,domain_hash;
};
using Supplied=std::shared_ptr<const Image>;
inline uint64_t hash(std::span<const uint8_t> bytes){uint64_t h=14695981039346656037ull;for(auto b:bytes){h^=b;h*=1099511628211ull;}return h;}
inline uint64_t words(std::span<const float> x){return hash({reinterpret_cast<const uint8_t*>(x.data()),x.size_bytes()});}
inline Result<Supplied> supply(Identity id,std::span<const float>x,std::span<const float>norm,const std::string&norm_tensor,const readers18::Family&family){
 if(id.layer>=64||!id.generation||!id.boundary||id.token<1001||uint64_t(id.token)+id.batch>248320||id.stream>=32)return Error::owner;
 if(id.batch!=1&&id.batch!=8&&id.batch!=32)return Error::shape;
 if(x.size()!=size_t(id.batch)*5120)return Error::source;
 std::string expected="blk."+std::to_string(id.layer)+(id.region==Region::layer_input?".attn_norm.weight":id.region==Region::post_attention?".post_attention_norm.weight":"");
 if(expected=="blk."+std::to_string(id.layer)||norm.size()!=5120||norm_tensor!=expected)return Error::norm;
 auto f=readers18::family(id.region==Region::layer_input?"input":"ffn",id.layer,id.batch,id.region==Region::layer_input?3:2);
 auto valid=readers18::validate(family);auto*p=std::get_if<readers18::Family>(&f);if(!std::holds_alternative<unsigned>(valid)||!p||family.source_layer!=id.layer||family.batch!=id.batch||family.token!=id.token||family.readers!=p->readers)return Error::manifest;
 for(float v:x)if(!std::isfinite(v))return Error::nonfinite;for(float v:norm)if(!std::isfinite(v))return Error::nonfinite;
 for(unsigned a=0;a<id.batch;a++)for(unsigned b=0;b<a;b++)if(!std::memcmp(x.data()+a*5120,x.data()+b*5120,5120*4))return Error::duplicate_row;
 uint64_t domain[8]={id.layer,id.batch,id.token,id.stream,id.generation,id.boundary,uint64_t(id.region),uint64_t(family.readers.size())};
 try{return Supplied{new Image(id,family,norm_tensor,{x.begin(),x.end()},{norm.begin(),norm.end()},words(x),words(norm),hash({reinterpret_cast<const uint8_t*>(domain),sizeof(domain)}))};}catch(const std::bad_alloc&){return Error::allocation;}
}
inline bool intact(const Supplied&s){return s&&s->source_hash==words(s->x)&&s->norm_hash==words(s->norm);}
}
