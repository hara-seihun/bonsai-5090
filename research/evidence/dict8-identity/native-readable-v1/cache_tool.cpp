#include "dict8_cache.hpp"
#include "dict8_offline.hpp"
#include "dict8_model.hpp"
#include "gguf.h"
#include "platform.h"
#include <atomic>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#endif
using namespace h5;
namespace c=dict8::cache;
static bool cleanup_failed=false;
struct Staging {std::filesystem::path dir;bool held=false;~Staging(){if(!held)return;std::error_code ec;
#ifdef _WIN32
 for(std::filesystem::recursive_directory_iterator it(dir,ec),end;!ec&&it!=end;it.increment(ec))if(it->is_regular_file(ec)) {if(_chmod(it->path().string().c_str(),_S_IREAD|_S_IWRITE))cleanup_failed=true;}
 if(ec)cleanup_failed=true;
#endif
 std::filesystem::remove_all(dir,ec);if(ec)cleanup_failed=true;
}};

int fail(c::Error e){std::cout<<"{\"ok\":false,\"error\":\""<<c::name(e)<<"\",\"code\":"<<int(e)<<",\"gpu_work\":false}\n";return int(e);}
c::Result<std::vector<uint8_t>> encode(const GTensor& gate,const GTensor& up,int threads,bool check,uint64_t& checked) {
 const GTensor* ts[]={&gate,&up};for(auto* t:ts)if(t->type!=143||t->ne!=std::vector<uint64_t>{c::K,c::N/2}||t->bytes!=uint64_t(c::N/2)*c::K/128*28)return c::Error::ModelLayout;
 std::vector<uint8_t> image(c::layer_bytes);std::atomic<unsigned> next{0};std::atomic<bool> bad{false};std::atomic<uint64_t> count{0};
 auto work=[&]{int8_t trit[32][128];std::array<uint8_t,2048> native{};auto book=dict8::dictionary();uint64_t local_count=0;
  for(;;){unsigned g=next.fetch_add(1);if(g>=c::N/64)break;
   for(unsigned kb=0;kb<c::K/128;kb++) {
    for(unsigned arm=0;arm<2;arm++){for(unsigned r=0;r<32;r++)dict8::offline::decode_weight_trits(ts[arm]->data+((size_t(g*32+r)*(c::K/128)+kb)*28),trit[r]);dict8::offline::wf0(trit,native.data()+arm*1024);}
    auto packed=dict8::pack_pair(native.data(),native.data()+1024);if(std::holds_alternative<dict8::Error>(packed)){bad=true;continue;}
    const auto& bytes=std::get<std::array<uint8_t,1664>>(packed);std::memcpy(image.data()+(size_t(kb)*(c::N/32)+g*2)*832,bytes.data(),bytes.size());
    if(check)for(unsigned lane=0;lane<32;lane++)for(unsigned w=0;w<16;w++)for(unsigned arm=0;arm<2;arm++){auto key=dict8::unpack_key(bytes.data(),lane,w,arm);if(key>=6561||book[key]!=dict8::native_word(native.data()+arm*1024,lane,w))bad=true;else ++local_count;}
   }
  }
  count.fetch_add(local_count);
 };
 std::vector<std::thread> workers;try{for(int i=0;i<threads;i++)workers.emplace_back(work);}catch(const std::bad_alloc&){for(auto& t:workers)t.join();return c::Error::Allocation;}catch(const std::system_error&){for(auto& t:workers)t.join();return c::Error::Threads;}for(auto& t:workers)t.join();checked+=count.load();if(bad)return c::Error::Corrupt;return image;
}
int run(int argc,char** argv) {
 if(argc!=7){std::cerr<<"cache_tool prepare|verify|compare MODEL CACHE THREADS MAX_BYTES HEADROOM_BYTES\n";return fail(c::Error::Unset);}
 std::string mode=argv[1];if(mode!="prepare"&&mode!="verify"&&mode!="compare")return fail(c::Error::Unset);
 auto number=[](const char* s,uint64_t& n){n=0;if(!*s)return false;for(;*s;s++){if(*s<'0'||*s>'9'||n>(UINT64_MAX-9)/10)return false;n=n*10+(*s-'0');}return true;};
 uint64_t threads,budget,headroom;if(!number(argv[4],threads)||threads<1||threads>64||!number(argv[5],budget)||!number(argv[6],headroom))return fail(c::Error::Budget);
 std::filesystem::path path=argv[3];std::error_code ec;Staging staging{path.string()+".preparing",false};
 if(mode=="prepare"){
  if(budget<c::file_bytes)return fail(c::Error::Budget);
  auto free=std::filesystem::space(path.parent_path(),ec);if(ec||free.available<budget||free.available-budget<headroom)return fail(c::Error::Budget);
  if(std::filesystem::exists(path,ec))return fail(c::Error::Exists);
  if(!std::filesystem::create_directory(staging.dir,ec))return fail(ec?c::Error::Io:c::Error::Staged);staging.held=true;
 }
 uint64_t start=c::tick();auto opened_model=c::open_model(argv[2]);if(auto* e=std::get_if<c::Error>(&opened_model))return fail(*e);auto model=std::move(std::get<std::unique_ptr<c::ExactModel>>(opened_model));const auto& g=model->gguf();const auto& mc=model->costs();auto model_hash_ns=mc.identity_ns;auto source_open_ns=mc.source_open_ns;
 uint64_t encode_ns=0,write_ns=0,output_hash_ns=0,checked=0,compare_ns=0; c::Costs costs;
 std::filesystem::path tmp;
 if(mode=="prepare") {
  auto h=c::header();tmp=staging.dir/"cache.part";c::File out(std::fopen(tmp.string().c_str(),"wb"));if(!out)return fail(c::Error::Io);
  if(std::fwrite(h.data(),1,h.size(),out.get())!=h.size())return fail(c::Error::Io);
  for(uint32_t layer=0;layer<c::layers;layer++) {
   auto prefix="blk."+std::to_string(layer)+".";start=c::tick();auto r=encode(g.t(prefix+"ffn_gate.weight"),g.t(prefix+"ffn_up.weight"),int(threads),false,checked);encode_ns+=c::tick()-start;
   if(auto* e=std::get_if<c::Error>(&r))return fail(*e);auto& bytes=std::get<std::vector<uint8_t>>(r);
   start=c::tick();auto digest=c::hash(bytes.data(),bytes.size());output_hash_ns+=c::tick()-start;if(auto* e=std::get_if<c::Error>(&digest))return fail(*e);std::memcpy(h.data()+128+layer*32,std::get<c::Digest>(digest).data(),32);
   start=c::tick();if(std::fwrite(bytes.data(),1,bytes.size(),out.get())!=bytes.size())return fail(c::Error::Io);write_ns+=c::tick()-start;
  }
  start=c::tick();if(std::fseek(out.get(),0,SEEK_SET)||std::fwrite(h.data(),1,h.size(),out.get())!=h.size()||std::fflush(out.get()))return fail(c::Error::Io);
#ifdef _WIN32
  if(_commit(_fileno(out.get())))return fail(c::Error::Publication);
#else
  if(fsync(fileno(out.get())))return fail(c::Error::Publication);
#endif
  out.reset();
  write_ns+=c::tick()-start;
 }
 // The same Reader used by Engine is the entire recurring read/identity/hash path.
 auto opened=c::Reader::open((mode=="prepare"?tmp:path).string().c_str(),*model,model->identity());if(auto* e=std::get_if<c::Error>(&opened))return fail(*e);auto reader=std::move(std::get<std::unique_ptr<c::Reader>>(opened));
 for(uint32_t layer=0;layer<c::layers;layer++) {
  auto r=reader->layer(layer);if(auto* e=std::get_if<c::Error>(&r))return fail(*e);
  if(mode=="compare") {start=c::tick();auto prefix="blk."+std::to_string(layer)+".";auto encoded=encode(g.t(prefix+"ffn_gate.weight"),g.t(prefix+"ffn_up.weight"),int(threads),true,checked);compare_ns+=c::tick()-start;if(auto* e=std::get_if<c::Error>(&encoded))return fail(*e);if(std::memcmp(std::get<c::View>(r).data(),std::get<std::vector<uint8_t>>(encoded).data(),c::layer_bytes))return fail(c::Error::Corrupt);}
 }
 auto done=reader->finish();if(auto* e=std::get_if<c::Error>(&done))return fail(*e);costs=reader->costs;
 if(mode=="prepare"){start=c::tick();
#ifdef _WIN32
  if(_chmod(tmp.string().c_str(),_S_IREAD))return fail(c::Error::Publication);
  if(!MoveFileExA(tmp.string().c_str(),path.string().c_str(),MOVEFILE_WRITE_THROUGH))return fail(c::Error::Publication);
#else
  if(chmod(tmp.c_str(),0444)||link(tmp.c_str(),path.c_str()))return fail(c::Error::Publication);
  if(unlink(tmp.c_str()))return fail(c::Error::PublishedCleanupRequired);
  int dirfd=::open(path.parent_path().c_str(),O_DIRECTORY);if(dirfd<0)return fail(c::Error::PublishedDurabilityUnknown);int status=fsync(dirfd);close(dirfd);if(status)return fail(c::Error::PublishedDurabilityUnknown);
#endif
  write_ns+=c::tick()-start;
 }
 std::cout<<"{\"ok\":true,\"mode\":\""<<mode<<"\",\"file_bytes\":"<<c::file_bytes<<",\"payload_bytes\":"<<c::layer_bytes*c::layers<<",\"header_bytes\":"<<c::header_bytes<<",\"model_source_open_ns\":"<<source_open_ns<<",\"preflight_model_hash_ns\":"<<model_hash_ns<<",\"model_allocation_ns\":"<<mc.allocation_ns<<",\"model_read_copy_ns\":"<<mc.read_copy_ns<<",\"model_protection_ns\":"<<mc.protection_ns<<",\"model_parse_ns\":"<<mc.parse_ns<<",\"model_source_read_bytes\":"<<mc.source_read_bytes<<",\"model_copied_bytes\":"<<mc.copied_bytes<<",\"model_allocated_bytes\":"<<mc.allocated_bytes<<",\"model_identity_bytes\":"<<mc.identity_bytes<<",\"model_identity_scans\":"<<mc.identity_scans<<",\"model_mapped_bytes\":"<<mc.mapped_bytes<<",\"model_admission_ns\":"<<mc.admission_ns<<",\"model_storage\":"<<mc.storage<<",\"encode_ns\":"<<encode_ns<<",\"write_flush_publish_ns\":"<<write_ns<<",\"one_time_payload_hash_ns\":"<<output_hash_ns<<",\"compare_ns\":"<<compare_ns<<",\"checked_original_words\":"<<checked<<",\"recurring_model_bytes\":"<<costs.model_bytes<<",\"recurring_model_hash_ns\":"<<costs.model_hash_ns<<",\"recurring_header_open_ns\":"<<costs.open_header_ns<<",\"recurring_read_bytes\":"<<costs.read_bytes<<",\"recurring_read_ns\":"<<costs.read_ns<<",\"recurring_payload_hash_ns\":"<<costs.hash_ns<<",\"recurring_allocation_ns\":"<<costs.allocation_ns<<",\"recurring_finish_ns\":"<<costs.finish_ns<<",\"checked_layers\":"<<costs.checked_layers<<",\"gpu_work\":false,\"device_upload_ns\":null,\"native_cached_startup_ns\":null}\n";
 return 0;
}
int main(int argc,char** argv){
 std::ostringstream output;auto* sink=std::cout.rdbuf(output.rdbuf());int result=0;
 try{result=run(argc,argv);}catch(const std::bad_alloc&){result=fail(c::Error::Allocation);}catch(const std::exception& e){std::cerr<<e.what()<<"\n";result=fail(c::Error::Io);}
 std::cout.rdbuf(sink);
 if(cleanup_failed)return fail(result==0?c::Error::PublishedCleanupRequired:c::Error::StagedCleanupRequired);
 std::cout<<output.str();return result;
}
