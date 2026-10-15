#include "intrinsic.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
using namespace halo6000;
using json=nlohmann::json;
namespace fs=std::filesystem;
static int execution_calls=0;
namespace halo6000 {
DeviceModel::~DeviceModel() {}
Status DeviceModel::reset(int) {++execution_calls;return Error{ErrorCode::Unsupported,"validation test must not execute"};}
Status run_token(const DeviceModel&,DecodeWorkspace,int,int,int,int,cudaStream_t) {++execution_calls;return Error{ErrorCode::Unsupported,"validation test must not execute"};}
Status cuda_status(cudaError_t e,const char*op) {++execution_calls;return e==cudaSuccess?success():Status(Error{ErrorCode::Cuda,op});}
}
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 char pattern[]="/tmp/halo-intrinsic-test-XXXXXX";char*created=mkdtemp(pattern);if(!created)return 2;
 fs::path root(created);std::ofstream(root/"source")<<"source bytes";
 Provenance provenance;provenance.source_path=(root/"source").string();provenance.cache_path=(root/"cache").string();
 struct stat st{};stat(provenance.source_path.c_str(),&st);provenance.source={uint64_t(st.st_dev),uint64_t(st.st_ino),uint64_t(st.st_size),uint64_t(st.st_mtim.tv_sec)*1000000000+st.st_mtim.tv_nsec,uint64_t(st.st_ctim.tv_sec)*1000000000+st.st_ctim.tv_nsec};
 DeviceModel m;m.device_weights=reinterpret_cast<ModelWeights*>(1);m.state.slots=1;m.state.context=512;m.state.gdn=m.state.conv=reinterpret_cast<float*>(1);m.state.key=m.state.value=reinterpret_cast<uint16_t*>(1);m.state.positions=reinterpret_cast<int32_t*>(1);
 m.weights.head.n=Vocab;m.weights.head.k=Hidden;m.weights.embedding.n=Vocab;m.weights.embedding.k=Hidden;DecodeWorkspace w;w.context=512;w.logits=reinterpret_cast<float*>(1);
 json original;std::ifstream(argv[1])>>original;json report={{"validation_only",true},{"gpu_execution",false},{"checks",json::array()}};
 int failures=0;
 auto test=[&](const char*name,json spec,Provenance p,fs::path output){std::ofstream(root/"spec.json")<<spec.dump();auto s=dump_intrinsic(m,w,p,(root/"spec.json").string(),output.string(),188);bool passed=!s.ok()&&execution_calls==0;report["checks"].push_back({{"name",name},{"passed",passed},{"error",s.ok()?"unexpected success":s.error().message}});failures+=!passed;};
 json x=original;x["context"]=511;test("context mismatch",x,provenance,root/"new1");
 x=original;x["add_bos"]=true;test("BOS rejected",x,provenance,root/"new2");
 x=original;x["output_shape"]={16,256,Vocab};test("off-by-one rows rejected",x,provenance,root/"new3");
 x=original;x["config_id"]="wrong";test("manifest mismatch",x,provenance,root/"new4");
 auto p=provenance;p.source.size++;test("changed source identity",original,p,root/"new5");
 p=provenance;p.version++;test("cache version",original,p,root/"new6");
 fs::create_directory(root/"exists");test("refuse existing output",original,provenance,root/"exists");
 std::ifstream input(original["tokens_file"].get<std::string>(),std::ios::binary);std::string bytes(32768,'\0');input.read(bytes.data(),bytes.size());bytes[0]^=1;std::ofstream(root/"tokens",std::ios::binary).write(bytes.data(),bytes.size());
 x=original;x["tokens_file"]=(root/"tokens").string();test("token SHA mismatch",x,provenance,root/"new7");
 report["execution_calls"]=execution_calls;report["failures"]=failures;std::cout<<report.dump(2)<<'\n';fs::remove_all(root);return failures?1:0;
}
