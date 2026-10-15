#pragma once
#include "dict8_model.hpp"
namespace h5::dict8::cache {
class Reader {
 enum class State { Ready, Failed, Finished };
 std::shared_ptr<const void> model_lifetime_;
 std::vector<uint8_t> image_;uint32_t next_=0;State state_=State::Ready;
 Reader()=default;
 template<class T> Result<T> fail(Error e){state_=State::Failed;return e;}
public:
 Costs costs;
 static Result<std::unique_ptr<Reader>> open(const char* path,const ExactModel& model,const ExactModel::Identity& identity) try {
  if(!path||!*path)return Error::Unset;
  auto retained=model.retain(identity);if(auto* e=std::get_if<Error>(&retained))return *e;
  uint64_t start=tick();std::error_code ec;auto size=std::filesystem::file_size(path,ec);if(ec)return ec==std::errc::no_such_file_or_directory?Error::Missing:Error::Io;
  if(size!=file_bytes)return Error::Truncated;
  File f(std::fopen(path,"rb"));if(!f)return Error::Io;Header h{};if(std::fread(h.data(),1,h.size(),f.get())!=h.size())return std::feof(f.get())?Error::Truncated:Error::Io;
  auto v=validate_header(h);if(auto* e=std::get_if<Error>(&v))return *e;
  auto r=std::unique_ptr<Reader>(new Reader);r->costs.open_header_ns=tick()-start;start=tick();
  r->model_lifetime_=std::move(std::get<std::shared_ptr<const void>>(retained));
  r->costs.model_bytes=model.costs().identity_bytes;
  r->image_.resize(layers*layer_bytes);r->costs.allocation_ns=tick()-start;
  for(uint32_t i=0;i<layers;i++) {
   auto* dst=r->image_.data()+i*layer_bytes;start=tick();if(std::fread(dst,1,layer_bytes,f.get())!=layer_bytes)return std::feof(f.get())?Error::Truncated:Error::Io;
   r->costs.read_ns+=tick()-start;r->costs.read_bytes+=layer_bytes;start=tick();auto d=hash(dst,layer_bytes);if(auto* e=std::get_if<Error>(&d))return *e;
   r->costs.hash_ns+=tick()-start;if(std::memcmp(std::get<Digest>(d).data(),h.data()+128+i*32,32))return Error::Corrupt;++r->costs.checked_layers;
  }
  if(std::fgetc(f.get())!=EOF)return Error::Truncated;if(std::ferror(f.get()))return Error::Io;
  return r;
 } catch(const std::bad_alloc&){return Error::Allocation;}
 Result<View> layer(uint32_t index) {
  if(state_!=State::Ready)return Error::State;
  if(index!=next_||index>=layers)return fail<View>(Error::LayerOrder);
  ++next_;return View{image_.data()+index*layer_bytes,size_t(layer_bytes)};
 }
 Result<bool> finish(){if(state_!=State::Ready)return Error::State;if(next_!=layers)return fail<bool>(Error::LayerOrder);auto start=tick();std::vector<uint8_t>().swap(image_);state_=State::Finished;costs.finish_ns=tick()-start;return true;}
};
}
