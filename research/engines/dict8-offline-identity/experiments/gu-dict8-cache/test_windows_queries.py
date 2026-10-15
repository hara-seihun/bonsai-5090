"""CPU mocked Win32 calls around the exact generated storage_facts body, not live Windows evidence."""
from pathlib import Path
import hashlib,json,subprocess,tempfile
p=Path(__file__).resolve().parent;root=p.parents[1];model=(root/'src/dict8_model.hpp').read_text()
a=model.index(' static Result<StorageFacts> storage_facts(');b=model.index('\n#endif',a);body=model[a:b]
program=r'''#if defined(NDEBUG)
#error Dict8 CPU assertions must remain enabled
#endif
#include "dict8_model.hpp"
#include <cassert>
#include <cwchar>
using DWORD=uint32_t;using HANDLE=int;
constexpr DWORD FILE_TYPE_UNKNOWN=0,NO_ERROR=0,ERROR_INVALID_PARAMETER=87,FILE_NAME_NORMALIZED=0,VOLUME_NAME_GUID=1,MAX_PATH=260;
enum Info {FileRemoteProtocolInfo,FileAttributeTagInfo,FileStandardInfo};
struct FILE_REMOTE_PROTOCOL_INFO {int unused;};
struct FILE_ATTRIBUTE_TAG_INFO {DWORD FileAttributes,ReparseTag;};
struct FILE_STANDARD_INFO {bool Directory,DeletePending;struct {int64_t QuadPart;} EndOfFile;};
struct State {
 DWORD last_error=55,type=1,type_error=0,remote_error=87;bool remote_success=false,volume_success=true,path_success=true,tag_success=true,standard_success=true,grow=false;
 std::wstring path=L"\\\\?\\Volume{fixture}\\file.gguf";
 unsigned path_fills=0,drive_calls=0;
};
static State state;
static DWORD GetLastError(){return state.last_error;}
static void SetLastError(DWORD e){state.last_error=e;}
static DWORD GetFileType(HANDLE h){assert(h==71);if(state.type_error)state.last_error=state.type_error;return state.type;}
static bool GetFileInformationByHandleEx(HANDLE h,Info info,void* out,size_t){assert(h==71);
 if(info==FileRemoteProtocolInfo){state.last_error=state.remote_error;return state.remote_success;}
 if(info==FileAttributeTagInfo){auto* v=static_cast<FILE_ATTRIBUTE_TAG_INFO*>(out);v->FileAttributes=0x20;v->ReparseTag=0;return state.tag_success;}
 auto* v=static_cast<FILE_STANDARD_INFO*>(out);v->Directory=false;v->DeletePending=false;v->EndOfFile.QuadPart=h5::dict8::cache::model_size;return state.standard_success;
}
static bool GetVolumeInformationByHandleW(HANDLE h,void*,DWORD,void*,void*,void*,wchar_t* fs,DWORD){assert(h==71);std::wcscpy(fs,L"NTFS");return state.volume_success;}
static DWORD GetFinalPathNameByHandleW(HANDLE h,wchar_t* out,DWORD n,DWORD flags){assert(h==71&&flags==VOLUME_NAME_GUID);if(!state.path_success)return 0;
 if(!out)return state.grow?8:DWORD(state.path.size()+1);
 ++state.path_fills;if(n<=state.path.size())return DWORD(state.path.size()+1);
 std::wcscpy(out,state.path.c_str());return DWORD(state.path.size());
}
static DWORD GetDriveTypeW(const wchar_t* root){++state.drive_calls;assert(std::wstring(root)==L"\\\\?\\Volume{fixture}\\");return 3;}
namespace h5::dict8::cache {struct Probe {
'''+body+r'''
};}
namespace c=h5::dict8::cache;
template<class T> void error(c::Result<T> r,c::Error expected){assert(std::holds_alternative<c::Error>(r)&&std::get<c::Error>(r)==expected);}
int main(){unsigned cases=0;
 auto collect=[](){return c::Probe::storage_facts(71);};
 auto r=collect();assert(std::get<bool>(c::admit_storage(std::get<c::StorageFacts>(r))));++cases;
 state={};state.type=0;r=collect();assert(std::holds_alternative<c::StorageFacts>(r));error(c::admit_storage(std::get<c::StorageFacts>(r)),c::Error::UnsupportedStorage);++cases;
 state={};state.type=0;state.type_error=5;error(collect(),c::Error::AdmissionQuery);++cases;
 state={};state.remote_success=true;error(collect(),c::Error::RemoteStorage);assert(state.path_fills==0);++cases;
 for(DWORD e:{1U,5U,50U,122U}){state={};state.remote_error=e;error(collect(),c::Error::AdmissionQuery);++cases;}
 state={};state.volume_success=false;error(collect(),c::Error::AdmissionQuery);++cases;
 state={};state.path_success=false;error(collect(),c::Error::AdmissionQuery);++cases;
 state={};state.grow=true;r=collect();assert(std::get<bool>(c::admit_storage(std::get<c::StorageFacts>(r)))&&state.path_fills==2);++cases;
 state={};state.path=L"no-guid-root";r=collect();assert(state.drive_calls==0);error(c::admit_storage(std::get<c::StorageFacts>(r)),c::Error::UnsupportedStorage);++cases;
 state={};state.tag_success=false;error(collect(),c::Error::AdmissionQuery);++cases;
 state={};state.standard_success=false;error(collect(),c::Error::AdmissionQuery);++cases;
 std::cout<<cases;
}
'''
# This test mocks Win32 APIs and compiles their generated caller on Linux.
program=program.replace('#include <cwchar>','#include <cwchar>\n#include <iostream>')
with tempfile.TemporaryDirectory(prefix='dict8-win-query-') as td:
 t=Path(td);(t/'query.cpp').write_text(program)
 command=['g++','-std=c++17','-O2','-I'+str(root/'src'),str(t/'query.cpp'),'-lcrypto','-o',str(t/'query')]
 r=subprocess.run(command,capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stdout+r.stderr
 r=subprocess.run([str(t/'query')],capture_output=True,text=True,timeout=5);assert r.returncode==0,r.stdout+r.stderr
 receipt={'extracted_storage_facts_sha256':hashlib.sha256(body.encode()).hexdigest(),'mocked_win32_caller_cases':int(r.stdout),'stale_error_cleared':True,'guid_buffer_growth':True,'missing_root_rejected':True,'same_handle_queries':True,'query_failure_errors':True,'proof':'Generated actual Windows storage_facts caller, injected Win32 functions compiled on Linux; not live remote/reparse-volume evidence','gpu_work':False}
 (p/'windows-query-test-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
