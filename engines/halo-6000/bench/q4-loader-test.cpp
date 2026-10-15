#include "cache.hpp"
#include "q4_0-format.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <random>
#include <unistd.h>
using namespace halo6000;
void units(){
 static_assert(CacheVersion==1);static_assert(uint32_t(WeightFormat::Q4_0)==8);
 HostTensor t;t.shape={512,2048,7};t.format=WeightFormat::Q4_0;auto w=t.view();assert(w.scale_group==32&&w.row_stride==288&&w.expert_stride==589824);
 assert(tensor_bytes(2,{2048,248320}).ok()&&tensor_bytes(2,{2048,248320}).value()==286064640);
 assert(tensor_bytes(2,{32,1}).ok()&&tensor_bytes(2,{32,1}).value()==18);assert(!tensor_bytes(2,{31,1}).ok());assert(!tensor_bytes(2,{UINT64_MAX,2}).ok());
 std::vector<uint8_t>src(65536*18),dst(src.size());std::mt19937 rng(81);
 for(auto&v:src)v=rng();for(int b=0;b<65536;b++){uint16_t d=b;memcpy(src.data()+b*18,&d,2);}
 assert(repack_q4_0_grouped128(src.data(),65536,dst.data(),dst.size()).ok());
 for(size_t b=0;b<65536;b++){assert(!memcmp(src.data()+b*18+2,dst.data()+(b/4)*72+(b%4)*16,16));assert(!memcmp(src.data()+b*18,dst.data()+(b/4)*72+64+(b%4)*2,2));}
 assert(!repack_q4_0_grouped128(src.data(),3,dst.data(),54).ok());assert(!repack_q4_0_grouped128(src.data(),65536,dst.data(),dst.size()-1).ok());assert(!repack_q4_0_grouped128(src.data(),65536,src.data(),src.size()).ok());
 std::vector<uint8_t>v;auto put=[&](auto x){size_t n=v.size();v.resize(n+sizeof(x));memcpy(v.data()+n,&x,sizeof(x));};std::string name="output.weight";put(uint32_t(0x46554747));put(uint32_t(3));put(uint64_t(1));put(uint64_t(0));put(uint64_t(name.size()));v.insert(v.end(),name.begin(),name.end());put(uint32_t(2));put(uint64_t(2048));put(uint64_t(1));put(uint32_t(2));put(uint64_t(0));v.resize((v.size()+31)&~size_t(31));v.resize(v.size()+1152);
 std::string p="/tmp/halo-q4-parser-"+std::to_string(getpid());{std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size());}
 auto g=Gguf::open(p);assert(g.ok()&&g.value().tensors.size()==1&&g.value().tensors[0].type==2&&g.value().tensors[0].bytes==1152);assert(!g.value().validate_model().ok());std::filesystem::remove(p);
 puts("{\"units_passed\":true,\"half_scale_bitpatterns\":65536,\"cache_version\":1,\"Q4_tag\":8}");
}
int head_fixture(const std::string&path){
 auto raw=MappedFile::open(path);if(!raw.ok()){fprintf(stderr,"%s\n",raw.error().message.c_str());return 1;}if(raw.value()->size!=286064640)return 1;
 std::vector<uint8_t>dst(raw.value()->size);auto s=repack_q4_0_grouped128(static_cast<const uint8_t*>(raw.value()->base),size_t(248320)*64,dst.data(),dst.size());if(!s.ok())return 1;
 const std::string p=path+".cache72.bin";{std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(dst.data()),dst.size());if(!f)return 1;}dst.clear();dst.shrink_to_fit();auto cached=MappedFile::open(p);if(!cached.ok())return 1;
 Gguf source;source.storage=raw.value();GgufTensor tensor;tensor.name="output.weight";tensor.shape={2048,248320};tensor.type=2;tensor.bytes=raw.value()->size;tensor.data=static_cast<const uint8_t*>(raw.value()->base);source.tensors.push_back(tensor);
 HostImage image;image.storage=cached.value();HostTensor t;t.name=tensor.name;t.shape=tensor.shape;t.source_type=2;t.format=WeightFormat::Q4_0;t.bytes=cached.value()->size;t.data=static_cast<const uint8_t*>(cached.value()->base);image.tensors.push_back(t);
 auto result=verify_image(source,image);if(!result.ok()){fprintf(stderr,"%s\n",result.error().message.c_str());return 2;}image.tensors[0].format=WeightFormat::F32;assert(!verify_image(source,image).ok());
 puts("{\"actual_head_loader_verify_passed\":true,\"values\":508559360,\"source_bytes\":286064640,\"no_requantization\":true}");return 0;
}
int main(int argc,char**argv){units();if(argc==1)return 0;
 if(argc==3&&std::string(argv[1])=="--head-fixture")return head_fixture(argv[2]);
 if(argc==4&&std::string(argv[1])=="--existing"){
  auto image=load_cached(argv[2],argv[3],false,false);if(!image.ok()){fprintf(stderr,"%s\n",image.error().message.c_str());return 1;}assert(image.value().provenance.version==1&&image.value().tensors.size()==733);auto*t=image.value().find("output.weight");assert(t&&t->format==WeightFormat::Q8Grouped128);printf("{\"existing_cache_unchanged\":true,\"cache\":\"%s\",\"tensors\":733,\"version\":1}\n",image.value().provenance.cache_path.c_str());return 0;
 }
 if(argc==4&&std::string(argv[1])=="--model"){
  auto g=Gguf::open(argv[2]);if(!g.ok()){fprintf(stderr,"%s\n",g.error().message.c_str());return 1;}auto valid=g.value().validate_model();if(!valid.ok()){fprintf(stderr,"%s\n",valid.error().message.c_str());return 1;}
  auto image=load_cached(argv[2],argv[3],true,true);if(!image.ok()){fprintf(stderr,"%s\n",image.error().message.c_str());return 1;}auto*t=image.value().find("output.weight");assert(t&&t->format==WeightFormat::Q4_0&&t->bytes==286064640);auto proof=verify_image(g.value(),image.value());if(!proof.ok()){fprintf(stderr,"%s\n",proof.error().message.c_str());return 2;}printf("{\"full_Q4_model_cache_verified\":true,\"cache\":\"%s\",\"tensors\":%zu,\"version\":1,\"head_bytes\":%zu}\n",image.value().provenance.cache_path.c_str(),image.value().tensors.size(),t->bytes);return 0;
 }
 fprintf(stderr,"usage: q4-loader-test [--head-fixture RAW | --existing SOURCE CACHE_DIR | --model SOURCE CACHE_DIR]\n");return 1;
}
