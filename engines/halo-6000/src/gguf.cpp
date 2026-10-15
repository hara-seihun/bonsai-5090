#include "gguf.hpp"
#include <cstring>
#include <cerrno>
#include <limits>
#include <algorithm>
#include <cmath>
#include <set>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
namespace halo6000 {
namespace {
Error error(ErrorCode code,const std::string&s){return {code,s};}
size_t scalar(uint32_t t){switch(t){case 0:case 1:case 7:return 1;case 2:case 3:return 2;case 4:case 5:case 6:return 4;case 10:case 11:case 12:return 8;default:return 0;}}
bool product(uint64_t a,uint64_t b,uint64_t&v){if(b&&a>std::numeric_limits<uint64_t>::max()/b)return false;v=a*b;return true;}
struct Reader {
 const uint8_t*p,*end;std::string fault;
 size_t left()const{return size_t(end-p);}bool ok()const{return fault.empty();}
 void fail(const std::string&s){if(ok())fault=s;}
 template<class T>T read(){T v{};if(!ok())return v;if(sizeof(T)>left()){fail("truncated scalar");return v;}memcpy(&v,p,sizeof(T));p+=sizeof(T);return v;}
 std::string string(){uint64_t n=read<uint64_t>();if(!ok())return {};if(n>left()){fail("truncated string");return {};}std::string s(reinterpret_cast<const char*>(p),size_t(n));p+=n;return s;}
 void bytes(std::vector<uint8_t>&v,uint64_t n){if(!ok())return;if(n>left()){fail("truncated array");return;}v.assign(p,p+n);p+=n;}
};
}
MappedFile::~MappedFile(){if(base)munmap(base,size);}
void MappedFile::discard(const void*data,size_t bytes)const{if(!bytes)return;size_t page=size_t(sysconf(_SC_PAGESIZE));uintptr_t start=reinterpret_cast<uintptr_t>(data)&~uintptr_t(page-1);uintptr_t end=(reinterpret_cast<uintptr_t>(data)+bytes+page-1)&~uintptr_t(page-1);madvise(reinterpret_cast<void*>(start),end-start,MADV_DONTNEED);}
Result<std::shared_ptr<MappedFile>>MappedFile::open(const std::string&path){
 int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC);if(fd<0)return error(ErrorCode::Io,"open "+path+": "+strerror(errno));
 struct stat st{};if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<=0){close(fd);return error(ErrorCode::Io,"invalid regular file "+path);}
 auto m=std::make_shared<MappedFile>();m->size=st.st_size;m->identity={uint64_t(st.st_dev),uint64_t(st.st_ino),uint64_t(st.st_size),uint64_t(st.st_mtim.tv_sec)*1000000000+uint64_t(st.st_mtim.tv_nsec),uint64_t(st.st_ctim.tv_sec)*1000000000+uint64_t(st.st_ctim.tv_nsec)};
 m->base=mmap(nullptr,m->size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);if(m->base==MAP_FAILED){m->base=nullptr;return error(ErrorCode::Io,"mmap "+path+": "+strerror(errno));}madvise(m->base,m->size,MADV_SEQUENTIAL);return m;
}
std::optional<int64_t>GgufValue::integer()const{
 if(raw.size()!=scalar(type))return {};
 switch(type){case 0:case 7:return raw[0];case 1:return int8_t(raw[0]);case 2:{uint16_t v;memcpy(&v,raw.data(),2);return v;}case 3:{int16_t v;memcpy(&v,raw.data(),2);return v;}case 4:{uint32_t v;memcpy(&v,raw.data(),4);return v;}case 5:{int32_t v;memcpy(&v,raw.data(),4);return v;}case 10:{uint64_t v;memcpy(&v,raw.data(),8);if(v>uint64_t(INT64_MAX))return {};return int64_t(v);}case 11:{int64_t v;memcpy(&v,raw.data(),8);return v;}default:return {};}
}
std::optional<double>GgufValue::number()const{if(type==6&&raw.size()==4){float v;memcpy(&v,raw.data(),4);return v;}if(type==12&&raw.size()==8){double v;memcpy(&v,raw.data(),8);return v;}auto i=integer();return i?std::optional<double>(*i):std::nullopt;}
Result<size_t>tensor_bytes(uint32_t type,const std::vector<uint64_t>&shape){
 if(shape.empty()||shape.size()>4)return error(ErrorCode::Invalid,"tensor dimension count");
 uint64_t n=1;for(auto d:shape)if(!d||!product(n,d,n))return error(ErrorCode::Invalid,"tensor dimension overflow");uint64_t bytes=0;
 if(type==0){if(!product(n,4,bytes))return error(ErrorCode::Invalid,"tensor byte overflow");}
 else if(type==1||type==30){if(!product(n,2,bytes))return error(ErrorCode::Invalid,"tensor byte overflow");}
 else if(type==2||type==8||type==143||type==10){int group=(type==2||type==8)?32:type==10?256:128,block=type==2?18:type==8?34:type==10?84:28;if(shape[0]%group)return error(ErrorCode::Invalid,"quantized row not group-aligned");if(!product(n/group,block,bytes))return error(ErrorCode::Invalid,"tensor byte overflow");}
 else return error(ErrorCode::Unsupported,"GGUF tensor type "+std::to_string(type));
 if(bytes>SIZE_MAX)return error(ErrorCode::Invalid,"tensor too large");
 return size_t(bytes);
}
Result<Gguf>Gguf::open(const std::string&path){auto mapped=MappedFile::open(path);if(!mapped.ok())return mapped.error();Gguf g;g.storage=std::move(mapped.value());Reader r{static_cast<const uint8_t*>(g.storage->base),static_cast<const uint8_t*>(g.storage->base)+g.storage->size,{}};
 if(r.read<uint32_t>()!=0x46554747)return error(ErrorCode::Invalid,"not GGUF");
 auto version=r.read<uint32_t>();if(version!=2&&version!=3)return error(ErrorCode::Unsupported,"GGUF version");uint64_t nt=r.read<uint64_t>(),nk=r.read<uint64_t>();if(!r.ok()||nt>r.left()/24||nk>r.left()/12)return error(ErrorCode::Truncated,"invalid GGUF counts");
 for(uint64_t i=0;i<nk&&r.ok();i++){auto name=r.string();GgufValue v;v.type=r.read<uint32_t>();v.count=1;
  if(v.type==8)v.strings.push_back(r.string());
  else if(v.type==9){v.element_type=r.read<uint32_t>();v.count=r.read<uint64_t>();if(v.element_type==8){if(v.count>r.left()/8){r.fail("array string count");break;}for(uint64_t j=0;j<v.count&&r.ok();j++)v.strings.push_back(r.string());}else{uint64_t bytes=0;size_t sz=scalar(v.element_type);if(!sz||!product(sz,v.count,bytes)){r.fail("array type/overflow");break;}r.bytes(v.raw,bytes);}}
  else{size_t sz=scalar(v.type);if(!sz){r.fail("metadata type");break;}r.bytes(v.raw,sz);}
  if(!g.metadata.emplace(std::move(name),std::move(v)).second)r.fail("duplicate metadata key");
 }
 if(!r.ok())return error(ErrorCode::Truncated,r.fault);
 std::map<std::string,bool>names;
 for(uint64_t i=0;i<nt&&r.ok();i++){GgufTensor t;t.name=r.string();uint32_t nd=r.read<uint32_t>();if(nd<1||nd>4){r.fail("tensor dimension count");break;}for(uint32_t j=0;j<nd;j++)t.shape.push_back(r.read<uint64_t>());t.type=r.read<uint32_t>();t.offset=r.read<uint64_t>();if(!r.ok())break;auto bytes=tensor_bytes(t.type,t.shape);if(!bytes.ok())return bytes.error();t.bytes=bytes.value();if(!names.emplace(t.name,true).second)return error(ErrorCode::Invalid,"duplicate tensor "+t.name);g.tensors.push_back(std::move(t));}
 if(!r.ok())return error(ErrorCode::Truncated,r.fault);
 uint64_t alignment=32;auto it=g.metadata.find("general.alignment");if(it!=g.metadata.end()){auto a=it->second.integer();if(!a||*a<=0)return error(ErrorCode::Invalid,"alignment");alignment=*a;}if(alignment>1048576||(alignment&(alignment-1)))return error(ErrorCode::Invalid,"alignment is not a bounded power of two");
 uint64_t pos=uint64_t(r.p-static_cast<const uint8_t*>(g.storage->base));if(pos>UINT64_MAX-alignment+1)return error(ErrorCode::Invalid,"data offset overflow");uint64_t data=(pos+alignment-1)&~(alignment-1);if(data>g.storage->size)return error(ErrorCode::Truncated,"missing data section");std::vector<std::pair<uint64_t,uint64_t>>ranges;
 for(auto&t:g.tensors){if(t.offset%alignment||t.offset>g.storage->size-data||t.bytes>g.storage->size-data-t.offset)return error(ErrorCode::Truncated,"tensor extent "+t.name);t.data=static_cast<const uint8_t*>(g.storage->base)+data+t.offset;ranges.emplace_back(t.offset,t.offset+t.bytes);}
 std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();i++)if(ranges[i].first<ranges[i-1].second)return error(ErrorCode::Invalid,"overlapping tensors");return g;
}
const GgufTensor*Gguf::find(const std::string&name)const{for(auto&t:tensors)if(t.name==name)return &t;return nullptr;}
Status Gguf::validate_model()const{
 auto a=metadata.find("general.architecture");if(a==metadata.end()||a->second.type!=8||a->second.strings.size()!=1||a->second.strings[0]!="qwen35moe")return error(ErrorCode::Geometry,"architecture must be qwen35moe");
 const std::pair<const char*,int>values[]={{"block_count",Layers},{"embedding_length",Hidden},{"attention.head_count",QueryHeads},{"attention.head_count_kv",KVHeads},{"attention.key_length",HeadDim},{"attention.value_length",HeadDim},{"expert_count",Experts},{"expert_used_count",TopExperts},{"expert_feed_forward_length",ExpertHidden},{"expert_shared_feed_forward_length",SharedHidden},{"ssm.conv_kernel",ConvTaps},{"ssm.state_size",StateDim},{"ssm.group_count",KeyHeads},{"ssm.time_step_rank",ValueHeads},{"ssm.inner_size",Inner},{"full_attention_interval",FullInterval},{"rope.dimension_count",RotaryDim}};
 for(auto&v:values){auto it=metadata.find(std::string("qwen35moe.")+v.first);if(it==metadata.end()||it->second.integer()!=std::optional<int64_t>(v.second))return error(ErrorCode::Geometry,std::string("metadata ")+v.first);}
 for(auto&v:std::vector<std::pair<std::string,double>>{{"rope.freq_base",RopeBase},{"attention.layer_norm_rms_epsilon",NormEpsilon}}){auto it=metadata.find("qwen35moe."+v.first);auto n=it==metadata.end()?std::optional<double>{}:it->second.number();if(!n||*n!=v.second)return error(ErrorCode::Geometry,"metadata "+v.first);}
 auto tok=metadata.find("tokenizer.ggml.tokens");if(tok==metadata.end()||tok->second.type!=9||tok->second.element_type!=8||tok->second.count!=Vocab||tok->second.strings.size()!=Vocab)return error(ErrorCode::Geometry,"vocabulary count");
 std::map<std::string,std::vector<uint64_t>>expected={{"token_embd.weight",{Hidden,Vocab}},{"output.weight",{Hidden,Vocab}},{"output_norm.weight",{Hidden}}};
 for(int l=0;l<Layers;l++){std::string p="blk."+std::to_string(l)+".";auto add=[&](const std::string&n,std::vector<uint64_t>s){expected[p+n]=std::move(s);};add("attn_norm.weight",{Hidden});add("post_attention_norm.weight",{Hidden});
  for(auto n:{"gate","up"}){add(std::string("ffn_")+n+"_exps.weight",{Hidden,ExpertHidden,Experts});add(std::string("ffn_")+n+"_shexp.weight",{Hidden,SharedHidden});}add("ffn_down_exps.weight",{ExpertHidden,Hidden,Experts});add("ffn_down_shexp.weight",{SharedHidden,Hidden});add("ffn_gate_inp.weight",{Hidden,Experts});add("ffn_gate_inp_shexp.weight",{Hidden});
  if(attention_layer(l)){add("attn_q.weight",{Hidden,QueryHeads*HeadDim*2});add("attn_k.weight",{Hidden,KVHeads*HeadDim});add("attn_v.weight",{Hidden,KVHeads*HeadDim});add("attn_output.weight",{QueryHeads*HeadDim,Hidden});add("attn_q_norm.weight",{HeadDim});add("attn_k_norm.weight",{HeadDim});}
  else{add("attn_qkv.weight",{Hidden,ConvChannels});add("attn_gate.weight",{Hidden,Inner});add("ssm_alpha.weight",{Hidden,ValueHeads});add("ssm_beta.weight",{Hidden,ValueHeads});add("ssm_out.weight",{Inner,Hidden});add("ssm_a",{ValueHeads});add("ssm_dt.bias",{ValueHeads});add("ssm_norm.weight",{StateDim});add("ssm_conv1d.weight",{ConvTaps,ConvChannels});}
 }
 const std::string prefix="halo6000.rotation.";
 std::set<std::string>rotated;
 for(const auto&entry:metadata){
  if(entry.first=="halo6000.rotation")return error(ErrorCode::Geometry,"rotation metadata requires a tensor and field");
  if(entry.first.compare(0,prefix.size(),prefix)!=0)continue;
  auto dot=entry.first.find_last_of('.');
  if(dot<prefix.size())return error(ErrorCode::Geometry,"rotation metadata requires a tensor and field");
  const std::string field=entry.first.substr(dot+1),matrix=entry.first.substr(prefix.size(),dot-prefix.size());
  if(matrix.empty()||(field!="block_size"&&field!="order"&&field!="signs_tensor"))return error(ErrorCode::Geometry,"unknown rotation metadata "+entry.first);
  rotated.insert(matrix);
 }
 for(const auto&matrix:rotated){
  auto target=expected.find(matrix);
  if(target==expected.end()||target->second.size()<2||(matrix.find("ffn_down")==std::string::npos&&matrix.find("ssm_out.weight")==std::string::npos&&matrix.find("attn_output.weight")==std::string::npos))return error(ErrorCode::Geometry,"unsupported rotation target "+matrix);
  auto block_entry=metadata.find(prefix+matrix+".block_size");
  if(block_entry==metadata.end()||block_entry->second.type!=4)return error(ErrorCode::Geometry,"rotation block_size must be UINT32 "+matrix);
  auto block=block_entry->second.integer();if(!block||!supported_rotation_block(int(*block))||target->second[0]%*block)return error(ErrorCode::Geometry,"rotation block "+matrix);
  auto order=metadata.find(prefix+matrix+".order");if(order==metadata.end()||order->second.type!=8||order->second.strings!=std::vector<std::string>{"D_H"})return error(ErrorCode::Geometry,"rotation must be scalar STRING D_H "+matrix);
  auto sign=metadata.find(prefix+matrix+".signs_tensor");if(sign!=metadata.end()){
   if(sign->second.type!=8||sign->second.strings.size()!=1)return error(ErrorCode::Geometry,"rotation signs name must be STRING "+matrix);
   const std::string&signname=sign->second.strings[0];const auto*t=find(signname);
   if(!t||t->type!=0||t->shape!=std::vector<uint64_t>{target->second[0]}||t->bytes!=target->second[0]*4||!t->data)return error(ErrorCode::Geometry,"rotation signs must be F32[K] "+matrix);
   for(size_t i=0;i<target->second[0];i++){float s;memcpy(&s,t->data+i*4,4);if(s!=1.f&&s!=-1.f)return error(ErrorCode::Geometry,"rotation signs must be +/-1 "+matrix);}expected.emplace(signname,t->shape);
  }
 }
 if(tensors.size()!=expected.size())return error(ErrorCode::Geometry,"tensor count: expected "+std::to_string(expected.size())+", got "+std::to_string(tensors.size()));
 for(auto&t:tensors){auto it=expected.find(t.name);if(it==expected.end()||it->second!=t.shape)return error(ErrorCode::Geometry,"tensor shape/name "+t.name);if(t.type==143&&t.elements()/128%16)return error(ErrorCode::Geometry,"PANE tensor group count "+t.name);if(t.type==8&&t.shape[0]%128)return error(ErrorCode::Geometry,"Q8 tensor row alignment "+t.name);if(t.type==2&&(t.name!="output.weight"||t.shape[0]%128))return error(ErrorCode::Geometry,"Q4_0 currently requires output.weight with K128 alignment "+t.name);}
 return success();
}
} // namespace halo6000
