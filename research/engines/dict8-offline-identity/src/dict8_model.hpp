#pragma once
#include "dict8_common.hpp"
#include "gguf.h"
#include "platform.h"
#include <cerrno>
#include <limits>
namespace h5::dict8::cache {
static size_t tbytes(uint32_t t, uint64_t n) {
  switch (t) { case 0: return n * 4; case 1: case 30: return n * 2; case 143: return n / 128 * 28; case 8: return n / 32 * 34; case 2: return n / 32 * 18; }
  throw std::runtime_error("gguf: unsupported tensor type " + std::to_string(t));
}
enum class ModelStorage : uint64_t { WindowsRetainedLocalNtfs=1, LinuxPrivateSnapshot=2 };
struct ModelCosts {
 uint64_t source_open_ns=0,allocation_ns=0,read_copy_ns=0,protection_ns=0,identity_ns=0,parse_ns=0;
 uint64_t source_read_bytes=0,copied_bytes=0,allocated_bytes=0,identity_bytes=0,identity_scans=0;
 uint64_t mapped_bytes=0,admission_ns=0,storage=0;
};
struct StorageFacts {
 uint32_t file_type,remote_query_error;std::wstring filesystem,volume_path;
 uint32_t drive_type,attributes,reparse_tag;bool directory,delete_pending;int64_t bytes;
};
inline Result<bool> admit_storage(const StorageFacts& f) {
 if(f.file_type!=1)return Error::UnsupportedStorage;
 if(f.remote_query_error==0)return Error::RemoteStorage;
 if(f.remote_query_error!=87)return Error::AdmissionQuery;
 if(f.filesystem!=L"NTFS")return Error::UnsupportedStorage;
 const std::wstring prefix=L"\\\\?\\Volume{";
 if(f.volume_path.compare(0,prefix.size(),prefix)||f.volume_path.find(L"}\\",prefix.size())==std::wstring::npos||f.drive_type!=3)return Error::UnsupportedStorage;
 if((f.attributes&(0x400U|0x1000U|0x40000U|0x400000U))||f.reparse_tag)return Error::ReparseStorage;
 if(f.directory)return Error::ModelDirectory;if(f.delete_pending)return Error::ModelDeletePending;
 if(f.bytes<0||uint64_t(f.bytes)!=model_size)return Error::ModelIdentity;
 return true;
}
class ExactModel {
 struct Backing {
  void* data=nullptr;size_t bytes=0,allocated=0;ModelCosts costs;
#ifdef _WIN32
  HANDLE file=INVALID_HANDLE_VALUE,section=nullptr;
#endif
  Backing()=default;Backing(const Backing&)=delete;Backing& operator=(const Backing&)=delete;
  ~Backing(){
#ifdef _WIN32
   // The deny-write handle must outlive the last view and section, including failure paths.
   if(data)UnmapViewOfFile(data);if(section)CloseHandle(section);if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
#else
   if(data)munmap(data,allocated);
#endif
  }
 };
 std::shared_ptr<Backing> backing_;
 // parsed_.base stays null: only Backing owns/unmaps storage.
 Gguf parsed_;
 explicit ExactModel(std::shared_ptr<Backing> b):backing_(std::move(b)){}
 Result<bool> parse() try {
  const void* base=backing_->data;auto& kv=parsed_.kv;auto& tensors=parsed_.tensors;auto& index=parsed_.index;
  const uint8_t* p = (const uint8_t*)base;
  auto rd = [&](auto& v) { memcpy(&v, p, sizeof v); p += sizeof v; };
  auto str = [&]() { uint64_t n; rd(n); std::string s((const char*)p, n); p += n; return s; };
  uint32_t magic, ver; uint64_t nt, nk; rd(magic); rd(ver); rd(nt); rd(nk);
  if (magic != 0x46554747) throw std::runtime_error("not gguf");
  auto scalar = [&](uint32_t t, GValue& v) {
    switch (t) {
      case 0: { uint8_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 1: { int8_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 2: { uint16_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 3: { int16_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 4: { uint32_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 5: { int32_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 6: { float x; rd(x); v.nums.push_back(x); v.ints.push_back((int64_t)x); break; }
      case 7: { uint8_t x; rd(x); v.ints.push_back(x); v.nums.push_back(x); break; }
      case 10: { uint64_t x; rd(x); v.ints.push_back((int64_t)x); v.nums.push_back((double)x); break; }
      case 11: { int64_t x; rd(x); v.ints.push_back(x); v.nums.push_back((double)x); break; }
      case 12: { double x; rd(x); v.nums.push_back(x); v.ints.push_back((int64_t)x); break; }
      case 8: v.strs.push_back(str()); break;
      default: throw std::runtime_error("gguf kv type");
    }
  };
  for (uint64_t i = 0; i < nk; i++) {
    std::string k = str(); GValue v; rd(v.type);
    if (v.type == 9) { rd(v.etype); uint64_t n; rd(n); for (uint64_t j = 0; j < n; j++) scalar(v.etype, v); }
    else scalar(v.type, v);
    kv[k] = std::move(v);
  }
  for (uint64_t i = 0; i < nt; i++) {
    GTensor t; t.name = str(); uint32_t nd; rd(nd); uint64_t n = 1;
    for (uint32_t j = 0; j < nd; j++) { uint64_t d; rd(d); t.ne.push_back(d); n *= d; }
    rd(t.type); rd(t.off); t.bytes = tbytes(t.type, n);
    index[t.name] = tensors.size(); tensors.push_back(t);
  }
  uint64_t align = 32; if (kv.count("general.alignment")) align = kv["general.alignment"].ints[0];
  uint64_t data = ((uint64_t)(p - (const uint8_t*)base) + align - 1) / align * align;
  for (auto& t : tensors) t.data = (const uint8_t*)base + data + t.off;
  return true;
 }catch(const std::bad_alloc&){return Error::Allocation;}catch(const std::exception&){return Error::ModelLayout;}
#ifdef _WIN32
 static Result<StorageFacts> storage_facts(HANDLE file) {
  SetLastError(NO_ERROR);const auto type=GetFileType(file);if(type==FILE_TYPE_UNKNOWN&&GetLastError()!=NO_ERROR)return Error::AdmissionQuery;
  FILE_REMOTE_PROTOCOL_INFO remote{};const auto remote_ok=GetFileInformationByHandleEx(file,FileRemoteProtocolInfo,&remote,sizeof remote);const DWORD remote_error=remote_ok?0:GetLastError();
  if(remote_ok)return Error::RemoteStorage;if(remote_error!=ERROR_INVALID_PARAMETER)return Error::AdmissionQuery;
  wchar_t fs[MAX_PATH]{};if(!GetVolumeInformationByHandleW(file,nullptr,0,nullptr,nullptr,nullptr,fs,MAX_PATH))return Error::AdmissionQuery;
  DWORD n=GetFinalPathNameByHandleW(file,nullptr,0,FILE_NAME_NORMALIZED|VOLUME_NAME_GUID);if(!n)return Error::AdmissionQuery;
  std::vector<wchar_t> path(size_t(n)+1);DWORD got;
  for(;;){got=GetFinalPathNameByHandleW(file,path.data(),DWORD(path.size()),FILE_NAME_NORMALIZED|VOLUME_NAME_GUID);if(!got)return Error::AdmissionQuery;if(got<path.size())break;path.resize(size_t(got)+1);}
  std::wstring name(path.data(),got);const auto end=name.find(L"}\\");const auto root=end==std::wstring::npos?std::wstring():name.substr(0,end+2);
  FILE_ATTRIBUTE_TAG_INFO tag{};if(!GetFileInformationByHandleEx(file,FileAttributeTagInfo,&tag,sizeof tag))return Error::AdmissionQuery;
  FILE_STANDARD_INFO standard{};if(!GetFileInformationByHandleEx(file,FileStandardInfo,&standard,sizeof standard))return Error::AdmissionQuery;
  return StorageFacts{type,remote_error,fs,name,root.empty()?0:GetDriveTypeW(root.c_str()),tag.FileAttributes,tag.ReparseTag,bool(standard.Directory),bool(standard.DeletePending),standard.EndOfFile.QuadPart};
 }
#endif
public:
 class Identity {
  std::weak_ptr<Backing> backing_;
  explicit Identity(const std::shared_ptr<Backing>& b):backing_(b){}
  friend class ExactModel;
 public:
  Identity(const Identity&)=default;Identity& operator=(const Identity&)=default;
 };
 ExactModel(const ExactModel&)=delete;ExactModel& operator=(const ExactModel&)=delete;
 ExactModel(ExactModel&&)=delete;ExactModel& operator=(ExactModel&&)=delete;
 static Result<std::unique_ptr<ExactModel>> open(const char* path) try {
  if(!path||!*path)return Error::Unset;
  if(model_size>std::numeric_limits<size_t>::max())return Error::Allocation;
  auto b=std::make_shared<Backing>();auto start=tick();b->bytes=size_t(model_size);
#ifdef _WIN32
  int chars=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,nullptr,0);if(!chars)return Error::Io;
  std::vector<wchar_t> wide_path(size_t(chars),L'\0');if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path,-1,wide_path.data(),chars))return Error::Io;
  b->file=CreateFileW(wide_path.data(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
  if(b->file==INVALID_HANDLE_VALUE){auto e=GetLastError();if(e==ERROR_SHARING_VIOLATION)return Error::ModelInUse;return e==ERROR_FILE_NOT_FOUND||e==ERROR_PATH_NOT_FOUND?Error::Missing:Error::Io;}
  b->costs.source_open_ns=tick()-start;start=tick();auto facts=storage_facts(b->file);if(auto* e=std::get_if<Error>(&facts))return *e;
  auto admitted=admit_storage(std::get<StorageFacts>(facts));b->costs.admission_ns=tick()-start;
  if(auto* e=std::get_if<Error>(&admitted))return *e;start=tick();
  b->section=CreateFileMappingW(b->file,nullptr,PAGE_READONLY,0,0,nullptr);if(!b->section)return Error::Mapping;
  b->data=MapViewOfFile(b->section,FILE_MAP_READ,0,0,b->bytes);if(!b->data)return Error::Mapping;
  b->costs.allocation_ns=tick()-start;b->costs.mapped_bytes=b->bytes;b->costs.storage=uint64_t(ModelStorage::WindowsRetainedLocalNtfs);
#else
  struct Input {int fd;~Input(){if(fd>=0)close(fd);}} input{::open(path,O_RDONLY|O_CLOEXEC)};
  if(input.fd<0)return errno==ENOENT?Error::Missing:Error::Io;
  struct stat st;if(fstat(input.fd,&st))return Error::Io;
  if(st.st_size<0||uint64_t(st.st_size)!=model_size)return Error::ModelIdentity;
  auto page_result=sysconf(_SC_PAGESIZE);if(page_result<=0)return Error::Protection;const size_t page=size_t(page_result);
  b->costs.source_open_ns=tick()-start;
  if(model_size>std::numeric_limits<size_t>::max()-page)return Error::Allocation;
  b->allocated=(b->bytes+page-1)/page*page;start=tick();
  b->data=mmap(nullptr,b->allocated,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(b->data==MAP_FAILED){b->data=nullptr;return Error::Allocation;}
  b->costs.allocated_bytes=b->allocated;b->costs.allocation_ns=tick()-start;start=tick();
  for(size_t off=0;off<b->bytes;){auto chunk=size_t(std::min<uint64_t>(b->bytes-off,0x1000000));
   auto got=::read(input.fd,static_cast<uint8_t*>(b->data)+off,chunk);if(got<0){if(errno==EINTR)continue;return Error::Io;}
   if(!got)return Error::ModelIdentity;off+=size_t(got);b->costs.source_read_bytes+=size_t(got);
  }
  uint8_t extra;ssize_t got;do{got=::read(input.fd,&extra,1);}while(got<0&&errno==EINTR);if(got<0)return Error::Io;if(got)return Error::ModelIdentity;
  b->costs.copied_bytes=b->bytes;b->costs.read_copy_ns=tick()-start;start=tick();
  if(mprotect(b->data,b->allocated,PROT_READ))return Error::Protection;
  b->costs.protection_ns=tick()-start;b->costs.storage=uint64_t(ModelStorage::LinuxPrivateSnapshot);
#endif
  start=tick();auto digest=hash(b->data,b->bytes);++b->costs.identity_scans;b->costs.identity_bytes=b->bytes;b->costs.identity_ns=tick()-start;
  if(auto* e=std::get_if<Error>(&digest))return *e;
  if(std::get<Digest>(digest)!=unhex(model_sha256))return Error::ModelIdentity;
  auto model=std::unique_ptr<ExactModel>(new ExactModel(b));start=tick();auto parsed=model->parse();b->costs.parse_ns=tick()-start;
  if(auto* e=std::get_if<Error>(&parsed))return *e;return model;
 }catch(const std::bad_alloc&){return Error::Allocation;}
 Identity identity()const{return Identity(backing_);}
 Result<std::shared_ptr<const void>> retain(const Identity& id)const {
  auto b=id.backing_.lock();if(!b)return Error::StaleIdentity;
  if(b!=backing_)return Error::SubstitutedIdentity;
  return std::shared_ptr<const void>(std::move(b));
 }
 const Gguf& gguf()const{return parsed_;}
 const ModelCosts& costs()const{return backing_->costs;}
};
inline Result<std::unique_ptr<ExactModel>> open_model(const char* path){return ExactModel::open(path);}
}
