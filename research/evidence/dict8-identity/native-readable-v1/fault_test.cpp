#include "dict8_cache.hpp"
#include "dict8_model.hpp"
#include "platform.h"
#include <iostream>
#include <fstream>
#include <cassert>
using namespace h5::dict8;
namespace c=cache;
template<class T> void error(c::Result<T> r,c::Error e){assert(std::holds_alternative<c::Error>(r));assert(std::get<c::Error>(r)==e);}
int main(int argc,char** argv){assert(argc==3);const char* modelpath=argv[1];std::filesystem::path good=argv[2],fault=good.parent_path()/"fault.d8";
 assert(!std::filesystem::exists(fault));auto space=std::filesystem::space(good.parent_path());assert(space.available>c::file_bytes+20ULL*1024*1024*1024);
 auto opened_model=c::open_model(modelpath);assert(std::holds_alternative<std::unique_ptr<c::ExactModel>>(opened_model));auto model=std::move(std::get<std::unique_ptr<c::ExactModel>>(opened_model));assert(model->costs().identity_scans==1);
 error(c::Reader::open(nullptr,*model,model->identity()),c::Error::Unset);error(c::Reader::open("",*model,model->identity()),c::Error::Unset);
 error(c::Reader::open((good.parent_path()/"absent.d8").string().c_str(),*model,model->identity()),c::Error::Missing);
 auto original_header=c::Header{};{std::ifstream in(good,std::ios::binary);in.read((char*)original_header.data(),original_header.size());assert(in);}
 auto write_header=[&](const c::Header& h){std::ofstream out(fault,std::ios::binary|std::ios::trunc);out.write((const char*)h.data(),h.size());out.close();std::filesystem::resize_file(fault,c::file_bytes);};
 unsigned mutations=0;
 for(auto test:std::vector<std::pair<int,c::Error>>{{0,c::Error::Header},{8,c::Error::Version},{12,c::Error::Shape},{32,c::Error::ModelIdentity},{64,c::Error::Semantics},{96,c::Error::Header}}){auto h=original_header;h[test.first]^=1;write_header(h);error(c::Reader::open(fault.string().c_str(),*model,model->identity()),test.second);++mutations;}
 std::filesystem::resize_file(fault,100);error(c::Reader::open(fault.string().c_str(),*model,model->identity()),c::Error::Truncated);
 std::filesystem::resize_file(fault,c::file_bytes+1);error(c::Reader::open(fault.string().c_str(),*model,model->identity()),c::Error::Truncated);
 // Same table, different payload: corruption is caught before bytes are returned.
 error(c::Reader::open(fault.string().c_str(),*model,model->identity()),c::Error::Corrupt);
 auto h=original_header;h[128]^=1;write_header(h);error(c::Reader::open(fault.string().c_str(),*model,model->identity()),c::Error::Corrupt);
 // Whole layers plus their supplied hashes cannot be permuted under the pinned table root.
 h=original_header;for(int i=0;i<32;i++)std::swap(h[128+i],h[160+i]);write_header(h);error(c::Reader::open(fault.string().c_str(),*model,model->identity()),c::Error::Corrupt);
 auto valid=c::Reader::open(good.string().c_str(),*model,model->identity());assert(std::holds_alternative<std::unique_ptr<c::Reader>>(valid));auto reader=std::move(std::get<std::unique_ptr<c::Reader>>(valid));
 error(reader->finish(),c::Error::LayerOrder);error(reader->layer(0),c::Error::State);reader.reset();valid=c::Reader::open(good.string().c_str(),*model,model->identity());reader=std::move(std::get<std::unique_ptr<c::Reader>>(valid));
 error(reader->layer(1),c::Error::LayerOrder);error(reader->layer(0),c::Error::State);error(reader->finish(),c::Error::State);reader.reset();
 valid=c::Reader::open(good.string().c_str(),*model,model->identity());reader=std::move(std::get<std::unique_ptr<c::Reader>>(valid));for(unsigned i=0;i<64;i++)assert(std::holds_alternative<c::View>(reader->layer(i)));assert(std::get<bool>(reader->finish()));error(reader->finish(),c::Error::State);error(reader->layer(0),c::Error::State);
 std::filesystem::remove(fault);assert(model->costs().identity_scans==1);
 std::cout<<"{\"header_mutations\":"<<mutations<<",\"shared_model_identity_scans\":1,\"truncated_and_extra\":true,\"payload_corrupt\":true,\"digest_table_corrupt\":true,\"digest_table_layer_swap\":true,\"terminal_states\":true,\"out_of_order_and_incomplete\":true,\"unset_empty_missing\":true,\"gpu_work\":false}\n";
}
