#include "cache.hpp"
#include "q4_0-format.hpp"
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
namespace halo6000 {
namespace {
Error err(ErrorCode c,const std::string&s){return {c,s};}
constexpr uint64_t HashBasis=14695981039346656037ULL;
uint64_t hash(const uint8_t*p,size_t n,uint64_t h=HashBasis){for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ULL;}return h;}
size_t align256(size_t n){return (n+255)&~size_t(255);}
struct Record{HostTensor tensor;uint64_t offset=0,checksum=0;};
struct Writer {
 std::vector<uint8_t>data;
 template<class T>void put(T v){size_t p=data.size();data.resize(p+sizeof(T));memcpy(data.data()+p,&v,sizeof(T));}
 void string(const std::string&s){put<uint32_t>(s.size());data.insert(data.end(),s.begin(),s.end());}
};
std::vector<uint8_t>manifest(const Provenance&p,const std::vector<Record>&rs){Writer w;for(char c:std::string("HALO6C01"))w.put<char>(c);w.put<uint32_t>(CacheVersion);w.put<uint64_t>(GeometryTag);for(auto v:{p.source.device,p.source.inode,p.source.size,p.source.mtime_ns,p.source.ctime_ns})w.put<uint64_t>(v);w.string(p.source_path);w.put<uint32_t>(rs.size());for(auto&r:rs){w.string(r.tensor.name);w.put<uint32_t>(r.tensor.shape.size());for(auto d:r.tensor.shape)w.put<uint64_t>(d);w.put<uint32_t>(r.tensor.source_type);w.put<uint32_t>(uint32_t(r.tensor.format));w.put<uint64_t>(r.tensor.bytes);w.put<uint64_t>(r.offset);w.put<uint64_t>(r.checksum);}return w.data;}
struct Cursor {const uint8_t*p;size_t n;bool ok=true;template<class T>T get(){T v{};if(!ok||n<sizeof(T)){ok=false;return v;}memcpy(&v,p,sizeof(T));p+=sizeof(T);n-=sizeof(T);return v;}std::string string(){auto sz=get<uint32_t>();if(!ok||sz>n){ok=false;return {};}std::string s(reinterpret_cast<const char*>(p),sz);p+=sz;n-=sz;return s;}};
struct File {int fd=-1;std::string temporary;~File(){if(fd>=0)close(fd);if(!temporary.empty())unlink(temporary.c_str());}};
Status write_at(int fd,const uint8_t*p,size_t n,uint64_t off){while(n){ssize_t v=pwrite(fd,p,n,off);if(v<0&&errno==EINTR)continue;if(v<=0)return err(ErrorCode::Io,"cache write: "+std::string(strerror(errno)));p+=v;n-=v;off+=v;}return success();}
Result<std::string>canonical(const std::string&s){std::error_code ec;auto path=std::filesystem::canonical(s,ec);if(ec)return err(ErrorCode::Io,"canonical "+s+": "+ec.message());return path.string();}
std::string key(const std::string&path,const SourceIdentity&s){Writer w;w.string(path);for(auto v:{s.device,s.inode,s.size,s.mtime_ns,s.ctime_ns,GeometryTag})w.put<uint64_t>(v);w.put<uint32_t>(CacheVersion);w.string("PANE2-Q8GROUP128-F32");char text[17];snprintf(text,sizeof(text),"%016llx",(unsigned long long)hash(w.data.data(),w.data.size()));return text;}
Result<std::vector<Record>>records(const Gguf&g){std::vector<Record>rs;for(auto&t:g.tensors){Record r;r.tensor.name=t.name;r.tensor.shape=t.shape;r.tensor.source_type=t.type;uint64_t n=t.elements();if(t.type==143){r.tensor.format=WeightFormat::Pane2;r.tensor.bytes=pane_bytes(n/128);}else if(t.type==2){r.tensor.format=WeightFormat::Q4_0;r.tensor.bytes=n/128*Q4GroupBytes;}else if(t.type==8){r.tensor.format=WeightFormat::Q8Grouped128;r.tensor.bytes=n/128*136;}else if(t.type==10){r.tensor.format=WeightFormat::Q2K;r.tensor.bytes=n/256*84;}else{r.tensor.format=WeightFormat::F32;r.tensor.bytes=n*4;}rs.push_back(std::move(r));}return rs;}
Status build(const Gguf&g,const Provenance&p,const std::string&path){auto rr=records(g);if(!rr.ok())return rr.error();auto rs=std::move(rr.value());auto head=manifest(p,rs);size_t off=align256(head.size());for(auto&r:rs){r.offset=off;if(r.tensor.bytes>SIZE_MAX-off-255)return err(ErrorCode::Invalid,"cache extent overflow");off=align256(off+r.tensor.bytes);}
 if(off>10ULL*1024*1024*1024){struct statvfs fs{};auto dir=std::filesystem::path(path).parent_path();if(statvfs(dir.c_str(),&fs))return err(ErrorCode::Io,"cache free-space query");uint64_t free=uint64_t(fs.f_bavail)*fs.f_frsize;if(free<off+40ULL*1024*1024*1024)return err(ErrorCode::Io,"cache would leave less than 40GiB free");}
 File f;f.temporary=path+".tmp."+std::to_string(getpid())+"."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());f.fd=::open(f.temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);if(f.fd<0)return err(ErrorCode::Io,"create cache: "+std::string(strerror(errno)));if(ftruncate(f.fd,off))return err(ErrorCode::Io,"size cache");
 for(size_t i=0;i<rs.size();i++){auto&r=rs[i];auto&t=g.tensors[i];uint64_t checksum=HashBasis;
  size_t elements=t.elements(),units=t.type==10?elements/256:t.type==2||t.type==143||t.type==8?elements/128:elements,chunk=t.type==2||t.type==143||t.type==8||t.type==10?65536:1048576;
  size_t input_unit=t.type==2?72:t.type==143?28:t.type==8?136:t.type==10?84:t.type==0?4:2,output_unit=t.type==2?Q4GroupBytes:t.type==143?34:t.type==8?136:t.type==10?84:4;
  std::vector<uint8_t>buffer(chunk*output_unit);
  for(size_t start=0;start<units;start+=chunk){size_t count=std::min(chunk,units-start),bytes=count*output_unit;Status status=success();
   if(t.type==143)status=repack_ptq(t.data+start*input_unit,count,buffer.data(),bytes);
   else if(t.type==2)status=repack_q4_0_grouped128(t.data+start*input_unit,count*4,buffer.data(),bytes);
   else if(t.type==8)status=repack_q8(t.data+start*input_unit,count,buffer.data(),bytes);
   else if(t.type==10)memcpy(buffer.data(),t.data+start*input_unit,bytes);
   else status=convert_small(t.data+start*input_unit,t.type,count,buffer.data(),bytes);
   if(!status.ok())return status.error();
   checksum=hash(buffer.data(),bytes,checksum);auto written=write_at(f.fd,buffer.data(),bytes,r.offset+start*output_unit);if(!written.ok())return written.error();g.storage->discard(t.data+start*input_unit,count*input_unit);
  }r.checksum=checksum;fprintf(stderr,"cache %s %zu bytes\n",t.name.c_str(),r.tensor.bytes);
 }
 auto current=MappedFile::open(p.source_path);if(!current.ok())return current.error();if(!(current.value()->identity==p.source))return err(ErrorCode::CacheMismatch,"source changed during repack");head=manifest(p,rs);auto written=write_at(f.fd,head.data(),head.size(),0);if(!written.ok())return written.error();if(fsync(f.fd))return err(ErrorCode::Io,"fsync cache");if(rename(f.temporary.c_str(),path.c_str()))return err(ErrorCode::Io,"publish cache");f.temporary.clear();File directory;directory.fd=::open(std::filesystem::path(path).parent_path().c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);if(directory.fd<0||fsync(directory.fd))return err(ErrorCode::Io,"fsync cache directory");return success();
}
Result<HostImage>read_cache(const Gguf&g,const Provenance&provenance,const std::string&path,bool check){auto m=MappedFile::open(path);if(!m.ok())return m.error();Cursor c{static_cast<const uint8_t*>(m.value()->base),m.value()->size,true};std::string magic;for(int j=0;j<8;j++)magic+=c.get<char>();if(magic!="HALO6C01"||c.get<uint32_t>()!=CacheVersion||c.get<uint64_t>()!=GeometryTag)return err(ErrorCode::CacheMismatch,"cache version/geometry");SourceIdentity s;s.device=c.get<uint64_t>();s.inode=c.get<uint64_t>();s.size=c.get<uint64_t>();s.mtime_ns=c.get<uint64_t>();s.ctime_ns=c.get<uint64_t>();if(!(s==provenance.source)||c.string()!=provenance.source_path)return err(ErrorCode::CacheMismatch,"cache source identity");uint32_t nt=c.get<uint32_t>();if(nt!=g.tensors.size())return err(ErrorCode::CacheMismatch,"cache tensor count");auto expected=records(g);if(!expected.ok())return expected.error();auto rs=std::move(expected.value());HostImage image;image.provenance=provenance;image.provenance.cache_path=path;image.storage=std::move(m.value());
 for(auto&r:rs){auto name=c.string();uint32_t nd=c.get<uint32_t>();if(nd!=r.tensor.shape.size())return err(ErrorCode::CacheMismatch,"cache dimensions "+name);for(auto d:r.tensor.shape)if(c.get<uint64_t>()!=d)return err(ErrorCode::CacheMismatch,"cache shape "+name);uint32_t type=c.get<uint32_t>(),format=c.get<uint32_t>();uint64_t bytes=c.get<uint64_t>(),off=c.get<uint64_t>(),checksum=c.get<uint64_t>();if(!c.ok||name!=r.tensor.name||type!=r.tensor.source_type||format!=uint32_t(r.tensor.format)||bytes!=r.tensor.bytes||off%256||off>image.storage->size||bytes>image.storage->size-off)return err(ErrorCode::CacheMismatch,"cache descriptor/extent "+name);r.offset=off;r.checksum=checksum;r.tensor.data=static_cast<const uint8_t*>(image.storage->base)+off;image.tensors.push_back(r.tensor);}
 size_t first=align256(size_t(c.p-static_cast<const uint8_t*>(image.storage->base)));for(auto&r:rs){if(r.offset!=first)return err(ErrorCode::CacheMismatch,"cache payload ordering");first=align256(first+r.tensor.bytes);if(check&&hash(r.tensor.data,r.tensor.bytes)!=r.checksum)return err(ErrorCode::CacheMismatch,"cache checksum "+r.tensor.name);if(check)image.storage->discard(r.tensor.data,r.tensor.bytes);}if(first!=image.storage->size)return err(ErrorCode::CacheMismatch,"cache file extent");return image;
}
}
uint32_t half_to_float_bits(uint16_t h){uint32_t sign=uint32_t(h&0x8000)<<16,e=(h>>10)&31,m=h&1023;if(e==31)return sign|0x7f800000|(m<<13);if(e)return sign|((e+112)<<23)|(m<<13);if(!m)return sign;int exponent=-14;while(!(m&1024)){m<<=1;exponent--;}return sign|uint32_t(exponent+127)<<23|(m&1023)<<13;}
Status repack_ptq(const uint8_t*src,size_t blocks,uint8_t*dest,size_t bytes){if(bytes!=pane_bytes(blocks))return err(ErrorCode::Invalid,"PANE output size");memset(dest,0,bytes);
 #pragma omp parallel for schedule(static)
 for(size_t b=0;b<blocks;b++){const uint8_t*p=src+b*28;uint32_t words[8]={};for(int m=0;m<26;m++){unsigned v=p[m];for(int t=0;t<(m<24?5:4);t++){unsigned w=v*3;v=w&255;unsigned tr=w>>8,code=tr==0?5:tr==2?1:0;int e=m<16?m+16*t:m<24?80+m-16+8*t:120+m-24+2*t,h=e/16,j=e%16;words[h]|=code<<(4*(j%8)+j/8);}}uint8_t*out=dest+(b/16)*544+(b%16)*32;memcpy(out,words,32);memcpy(dest+(b/16)*544+512+(b%16)*2,p+26,2);}return success();}
Status repack_q8(const uint8_t*src,size_t groups,uint8_t*dest,size_t bytes){if(bytes!=groups*136)return err(ErrorCode::Invalid,"Q8 output size");
 #pragma omp parallel for schedule(static)
 for(size_t g=0;g<groups;g++)for(int j=0;j<4;j++){memcpy(dest+g*136+j*32,src+g*136+j*34+2,32);memcpy(dest+g*136+128+j*2,src+g*136+j*34,2);}return success();}
Status convert_small(const uint8_t*src,uint32_t type,size_t n,uint8_t*dest,size_t bytes){if(bytes!=n*4||(type!=0&&type!=1&&type!=30))return err(ErrorCode::Unsupported,"small tensor conversion");if(type==0){memcpy(dest,src,bytes);return success();}
 #pragma omp parallel for schedule(static)
 for(size_t i=0;i<n;i++){uint16_t h;memcpy(&h,src+i*2,2);uint32_t v=type==1?half_to_float_bits(h):uint32_t(h)<<16;memcpy(dest+i*4,&v,4);}return success();}
Result<std::string>cache_path(const std::string&source,const std::string&directory){auto p=canonical(source);if(!p.ok())return p.error();auto m=MappedFile::open(p.value());if(!m.ok())return m.error();return (std::filesystem::path(directory)/("halo6000-"+key(p.value(),m.value()->identity)+".bin")).string();}
Result<HostImage>load_cached(const std::string&source,const std::string&directory,bool create,bool check){auto p=canonical(source);if(!p.ok())return p.error();auto gg=Gguf::open(p.value());if(!gg.ok())return gg.error();auto&g=gg.value();auto valid=g.validate_model();if(!valid.ok())return valid.error();Provenance provenance;provenance.source_path=p.value();provenance.source=g.storage->identity;std::string path=(std::filesystem::path(directory)/("halo6000-"+key(p.value(),provenance.source)+".bin")).string();std::error_code ec;if(!std::filesystem::exists(path,ec)){
 if(!create)return err(ErrorCode::Io,"cache missing "+path);
 std::filesystem::create_directories(directory,ec);if(ec)return err(ErrorCode::Io,"cache directory: "+ec.message());File lock;lock.fd=::open((path+".lock").c_str(),O_RDWR|O_CREAT|O_CLOEXEC,0600);if(lock.fd<0||flock(lock.fd,LOCK_EX))return err(ErrorCode::Io,"cache lock");if(!std::filesystem::exists(path,ec)){auto built=build(g,provenance,path);if(!built.ok())return built.error();}}
 return read_cache(g,provenance,path,check);
}
Status verify_image(const Gguf&source,const HostImage&image){if(source.tensors.size()!=image.tensors.size())return err(ErrorCode::CacheMismatch,"verify tensor count");
 auto expected_records=records(source);if(!expected_records.ok())return expected_records.error();
 for(size_t i=0;i<source.tensors.size();i++){auto&s=source.tensors[i];auto&t=image.tensors[i];const auto&expected_tensor=expected_records.value()[i].tensor;if(t.name!=s.name||t.shape!=s.shape||t.source_type!=s.type||t.format!=expected_tensor.format||t.bytes!=expected_tensor.bytes||!t.data)return err(ErrorCode::CacheMismatch,"verify descriptor");
  if(s.type==143){size_t nb=s.elements()/128;
   for(size_t b=0;b<nb;b++){const uint8_t*p=s.data+b*28;const uint32_t*w=pane_words(t.data,b);if(memcmp(pane_scale(t.data,b),p+26,2))return err(ErrorCode::CacheMismatch,"verify PANE scale "+s.name);for(int m=0;m<26;m++){unsigned v=p[m];for(int j=0;j<(m<24?5:4);j++){unsigned peel=v*3;v=peel&255;int tr=int(peel>>8)-1,e=m<16?m+16*j:m<24?80+m-16+8*j:120+m-24+2*j,h=e/16,z=e%16;unsigned code=(w[h]>>(4*(z%8)+z/8))&5;int recovered=code==5?-1:code==1?1:0;if(recovered!=tr)return err(ErrorCode::CacheMismatch,"verify PANE trit "+s.name);}}}
  }else if(s.type==2){for(size_t b=0;b<s.elements()/32;b++)if(memcmp(s.data+b*18+2,t.data+(b/4)*Q4GroupBytes+(b%4)*16,16)||memcmp(s.data+b*18,t.data+(b/4)*Q4GroupBytes+64+(b%4)*2,2))return err(ErrorCode::CacheMismatch,"verify Q4_0 "+s.name);}
  else if(s.type==10){if(s.bytes!=t.bytes||memcmp(s.data,t.data,s.bytes))return err(ErrorCode::CacheMismatch,"verify Q2K "+s.name);}
  else if(s.type==8){for(size_t g=0;g<s.elements()/128;g++)for(int q=0;q<4;q++)if(memcmp(s.data+g*136+q*34+2,t.data+g*136+q*32,32)||memcmp(s.data+g*136+q*34,t.data+g*136+128+q*2,2))return err(ErrorCode::CacheMismatch,"verify Q8 "+s.name);}
  else{for(size_t j=0;j<s.elements();j++){uint32_t expected;if(s.type==0)memcpy(&expected,s.data+j*4,4);else{uint16_t v;memcpy(&v,s.data+j*2,2);expected=s.type==1?half_to_float_bits(v):uint32_t(v)<<16;}uint32_t got;memcpy(&got,t.data+j*4,4);if(expected!=got)return err(ErrorCode::CacheMismatch,"verify F32 "+s.name);}}
  source.storage->discard(s.data,s.bytes);image.storage->discard(t.data,t.bytes);
 }return success();
}
} // namespace halo6000
