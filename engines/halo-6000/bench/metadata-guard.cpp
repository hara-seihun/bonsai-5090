#include "cache.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unistd.h>
using namespace halo6000;
namespace fs=std::filesystem;
void need(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
template<class T>T take(Result<T>r){if(!r.ok())throw std::runtime_error(r.error().message);return std::move(r.value());}
GgufValue text(const std::string&s){GgufValue v;v.type=8;v.count=1;v.strings={s};return v;}
template<class T>GgufValue numeric(uint32_t type,T n){GgufValue v;v.type=type;v.count=1;v.raw.resize(sizeof(n));memcpy(v.raw.data(),&n,sizeof(n));return v;}
void rotate(Gguf&g,const std::string&t,uint32_t block){g.metadata["halo6000.rotation."+t+".block_size"]=numeric(4,block);g.metadata["halo6000.rotation."+t+".order"]=text("D_H");}
struct Temp {fs::path dir;Temp(){dir=fs::temp_directory_path()/("halo-metadata-guard-"+std::to_string(getpid()));need(fs::create_directory(dir),"Temporary directory exists");}~Temp(){std::error_code ec;fs::remove_all(dir,ec);}};
void write(const fs::path&p,const std::vector<uint8_t>&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size());need(bool(f),"Fixture write failed");}
std::vector<uint8_t> wire_q2(const std::vector<uint8_t>&payload,uint64_t k=512,uint32_t type=10){
 std::vector<uint8_t>v;auto put=[&](auto n){auto p=v.size();v.resize(p+sizeof(n));memcpy(v.data()+p,&n,sizeof(n));};
 put(uint32_t(0x46554747));put(uint32_t(3));put(uint64_t(1));put(uint64_t(0));
 put(uint64_t(2));v.insert(v.end(),{'q','2'});put(uint32_t(3));put(k);put(uint64_t(256));put(uint64_t(8));put(type);put(uint64_t(0));
 v.resize((v.size()+31)&~size_t(31));v.insert(v.end(),payload.begin(),payload.end());return v;
}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: metadata-guard GEOMETRY.gguf");
 auto base=take(Gguf::open(argv[1]));need(base.validate_model().ok(),"Valid baseline geometry rejected");
 std::vector<std::string>rejected;int accepted=1;
 auto bad=[&](const std::string&label,const std::function<void(Gguf&)>&mutate){auto g=base;mutate(g);auto status=g.validate_model();need(!status.ok(),"Accepted malformed metadata: "+label);need(status.error().code==ErrorCode::Geometry,"Wrong error category: "+label);rejected.push_back(label);};
 const std::string down="blk.0.ffn_down_exps.weight",p="halo6000.rotation."+down;
 for(const auto&t:base.tensors)if(t.name.find("ffn_down")!=std::string::npos||t.name.find("ssm_out.weight")!=std::string::npos||t.name.find("attn_output.weight")!=std::string::npos){auto g=base;rotate(g,t.name,128);need(g.validate_model().ok(),"Allowed writer/down rotation rejected "+t.name);++accepted;}
 for(uint32_t b:{128u,256u,512u,4096u}){auto g=base;rotate(g,"blk.0.ssm_out.weight",b);need(g.validate_model().ok(),"Supported block rejected");++accepted;}
 bad("orphan_order",[&](auto&g){g.metadata[p+".order"]=text("D_H");});
 bad("orphan_signs",[&](auto&g){g.metadata[p+".signs_tensor"]=text("signs");});
 bad("missing_order",[&](auto&g){g.metadata[p+".block_size"]=numeric(4,uint32_t(128));});
 bad("extra_rotation_field",[&](auto&g){rotate(g,down,128);g.metadata[p+".extra"]=text("unused");});
 bad("missing_tensor_and_field",[&](auto&g){g.metadata["halo6000.rotation"]=text("unused");});
 bad("empty_rotation_suffix",[&](auto&g){g.metadata["halo6000.rotation."]=text("unused");});
 bad("missing_rotation_tensor",[&](auto&g){g.metadata["halo6000.rotation.block_size"]=numeric(4,uint32_t(128));});
 bad("empty_rotation_tensor",[&](auto&g){g.metadata["halo6000.rotation..block_size"]=numeric(4,uint32_t(128));});
 bad("unsupported_gate_target",[&](auto&g){rotate(g,"blk.0.ffn_gate_exps.weight",128);});
 bad("unsupported_norm_target",[&](auto&g){rotate(g,"output_norm.weight",128);});
 bad("unknown_target",[&](auto&g){rotate(g,"blk.99.ffn_down_exps.weight",128);});
 bad("block_not_supported",[&](auto&g){rotate(g,down,1024);});
 bad("block_not_divisible",[&](auto&g){rotate(g,down,4096);});
 bad("block_zero",[&](auto&g){rotate(g,down,0);});
 bad("block_uint32_max",[&](auto&g){rotate(g,down,UINT32_MAX);});
 bad("block_signed",[&](auto&g){rotate(g,down,128);g.metadata[p+".block_size"]=numeric(5,int32_t(128));});
 bad("block_negative",[&](auto&g){rotate(g,down,128);g.metadata[p+".block_size"]=numeric(5,int32_t(-128));});
 bad("block_bool",[&](auto&g){rotate(g,down,128);g.metadata[p+".block_size"]=numeric(7,uint8_t(128));});
 bad("block_uint64_narrowing",[&](auto&g){rotate(g,down,128);g.metadata[p+".block_size"]=numeric(10,uint64_t(0x100000080ULL));});
 bad("block_float",[&](auto&g){rotate(g,down,128);g.metadata[p+".block_size"]=numeric(6,float(128));});
 bad("order_H_D",[&](auto&g){rotate(g,down,128);g.metadata[p+".order"]=text("H_D");});
 bad("order_array",[&](auto&g){rotate(g,down,128);g.metadata[p+".order"].type=9;g.metadata[p+".order"].element_type=8;});
 bad("order_empty",[&](auto&g){rotate(g,down,128);g.metadata[p+".order"]=text("");});
 bad("architecture_array",[&](auto&g){g.metadata["general.architecture"].type=9;});
 bad("vocabulary_numeric_array",[&](auto&g){g.metadata["tokenizer.ggml.tokens"].element_type=4;});
 bad("vocabulary_missing_strings",[&](auto&g){g.metadata["tokenizer.ggml.tokens"].strings.clear();});
 std::vector<float>signs(512);for(size_t i=0;i<signs.size();++i)signs[i]=i%2?1.f:-1.f;
 auto add_signs=[&](Gguf&g){rotate(g,down,128);g.metadata[p+".signs_tensor"]=text("rotation_signs");GgufTensor t;t.name="rotation_signs";t.type=0;t.shape={512};t.bytes=2048;t.data=reinterpret_cast<const uint8_t*>(signs.data());g.tensors.push_back(t);};
 auto signed_base=base;add_signs(signed_base);need(signed_base.validate_model().ok(),"Valid signs rejected");++accepted;
 bad("missing_signs_tensor",[&](auto&g){rotate(g,down,128);g.metadata[p+".signs_tensor"]=text("missing");});
 bad("signs_metadata_array",[&](auto&g){add_signs(g);g.metadata[p+".signs_tensor"].type=9;});
 bad("signs_F16",[&](auto&g){add_signs(g);g.tensors.back().type=1;});
 bad("signs_wrong_K",[&](auto&g){add_signs(g);g.tensors.back().shape={256};});
 bad("signs_extra_dimension",[&](auto&g){add_signs(g);g.tensors.back().shape={512,1};});
 bad("signs_short_extent",[&](auto&g){add_signs(g);g.tensors.back().bytes=4;});
 bad("signs_null_data",[&](auto&g){add_signs(g);g.tensors.back().data=nullptr;});
 bad("signs_zero",[&](auto&g){add_signs(g);signs[0]=0;});signs[0]=-1;
 bad("signs_nan",[&](auto&g){add_signs(g);signs[0]=std::numeric_limits<float>::quiet_NaN();});signs[0]=-1;
 bad("signs_infinity",[&](auto&g){add_signs(g);signs[0]=std::numeric_limits<float>::infinity();});signs[0]=-1;
 bad("unreferenced_sign_tensor",[&](auto&g){add_signs(g);g.metadata.erase(p+".signs_tensor");});
 Temp temp;std::mt19937 rng(712);std::vector<uint8_t>q2(4096*84);for(size_t i=0;i<q2.size();++i)q2[i]=i<256?uint8_t(i):uint8_t(rng());
 write(temp.dir/"source.gguf",wire_q2(q2));auto source=take(Gguf::open((temp.dir/"source.gguf").string()));
 write(temp.dir/"cache.bytes",q2);HostImage image;image.storage=take(MappedFile::open((temp.dir/"cache.bytes").string()));HostTensor t;t.name="q2";t.shape={512,256,8};t.source_type=10;t.format=WeightFormat::Q2K;t.bytes=q2.size();t.data=static_cast<const uint8_t*>(image.storage->base);image.tensors={t};
 need(verify_image(source,image).ok(),"Native Q2 byte identity rejected");
 int cache_errors=0;
 for(size_t at:{size_t(0),size_t(16),size_t(80),size_t(82),q2.size()-1}){auto corrupt=q2;corrupt[at]^=1;write(temp.dir/"corrupt.bytes",corrupt);auto im=image;im.storage=take(MappedFile::open((temp.dir/"corrupt.bytes").string()));im.tensors[0].data=static_cast<const uint8_t*>(im.storage->base);need(!verify_image(source,im).ok(),"Corrupt Q2 byte accepted");++cache_errors;}
 for(int i=0;i<5;++i){auto im=image;switch(i){case 0:im.tensors[0].format=WeightFormat::F32;break;case 1:im.tensors[0].source_type=8;break;case 2:--im.tensors[0].bytes;break;case 3:im.tensors[0].shape[0]=256;break;case 4:im.tensors[0].data=nullptr;break;}need(!verify_image(source,im).ok(),"Bad Q2 descriptor accepted");++cache_errors;}
 int views=0;for(int k:{256,512,2048,4096}){HostTensor d;d.shape={uint64_t(k),512,8};d.format=WeightFormat::Q2K;auto v=d.view();need(v.scale_group==16&&v.row_stride==size_t(k)/256*84&&v.expert_stride==512*v.row_stride,"Q2 view stride mismatch");++views;}
 need(uint32_t(WeightFormat::Q2K)==7,"Q2 cache tag changed");need(CacheVersion==1,"Existing v1 cache version changed");
 need(!tensor_bytes(10,{128,256}).ok(),"Short Q2 row accepted");need(!tensor_bytes(10,{UINT64_MAX,256}).ok(),"Q2 overflow accepted");
 auto short_wire=wire_q2(q2);short_wire.pop_back();write(temp.dir/"short.gguf",short_wire);need(!Gguf::open((temp.dir/"short.gguf").string()).ok(),"Truncated Q2 extent accepted");
 write(temp.dir/"unsupported.gguf",wire_q2(q2,512,11));need(!Gguf::open((temp.dir/"unsupported.gguf").string()).ok(),"Unsupported Q3 source accepted");
 std::cout<<"{\"metadata_positive_cases\":"<<accepted<<",\"metadata_negative_cases\":"<<rejected.size()<<",\"q2_blocks_all_bytes_identical\":4096,\"q2_payload_bytes\":"<<q2.size()<<",\"q2_corrupt_payload_descriptor_rejections\":"<<cache_errors<<",\"q2_stride_cases\":"<<views<<",\"q2_parser_negative_cases\":4,\"inference\":false,\"negative_labels\":[";
 for(size_t i=0;i<rejected.size();++i){std::cout<<(i?",":"")<<'"'<<rejected[i]<<'"';}
 std::cout<<"]}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
