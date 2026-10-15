#pragma once
#include "../round17/controller.hpp"
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <poll.h>
#include <chrono>
#include <csignal>
namespace live17 {
inline volatile sig_atomic_t interrupted=0;
inline void stop(int){interrupted=1;}
inline bool write_all(int fd,const std::string&text){size_t at=0;while(at<text.size()){auto n=::write(fd,text.data()+at,text.size()-at);if(n<0&&errno==EINTR)continue;if(n<=0)return false;at+=size_t(n);}return true;}
class Journal {
 int fd_=-1;std::string directory_;
 public:
 explicit Journal(std::string directory):directory_(std::move(directory)){}
 ~Journal(){if(fd_>=0)::close(fd_);}Journal(const Journal&)=delete;
 bool create(){fd_=::open((directory_+"/requests.log").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);if(fd_<0)return false;int d=::open(directory_.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);if(d<0)return false;bool ok=::fsync(d)==0;::close(d);return ok;}
 bool record(const std::string&line){return fd_>=0&&write_all(fd_,line+"\n")&&::fsync(fd_)==0;}
 bool accepted(const Request&r){return record("A "+encode(r));}
 bool completed(const Request&r,const Reply&reply){return record("D "+encode(r)+" "+encode(reply));}
 const std::string&directory()const{return directory_;}
};
inline std::string render(const Lookup&lookup){if(auto p=std::get_if<Reply>(&lookup))return "D "+encode(*p);if(auto p=std::get_if<Pending>(&lookup))return "A "+encode(p->request);return "E "+std::to_string(unsigned(std::get<Error>(lookup)));}
template<class Backend>int serve(Backend&backend,Journal&journal,uint64_t epoch,size_t maximum,unsigned lifetime_seconds){
 if(!epoch||!maximum||maximum>32768||!lifetime_seconds||lifetime_seconds>3600)return 2;
 if(!journal.create()){std::puts("{\"state\":\"journal_exists_or_io_error_refused_context_recovery\",\"GPU_calls\":0}");return 3;}
 std::string path=journal.directory()+"/worker.sock";if(path.size()>=sizeof(sockaddr_un::sun_path))return 3;
 int server=::socket(AF_UNIX,SOCK_SEQPACKET|SOCK_CLOEXEC,0);if(server<0)return 3;sockaddr_un address{};address.sun_family=AF_UNIX;std::memcpy(address.sun_path,path.c_str(),path.size()+1);
 if(::bind(server,reinterpret_cast<sockaddr*>(&address),sizeof(address))||::listen(server,16)){::close(server);return 3;}
 ::signal(SIGTERM,stop);::signal(SIGINT,stop);Controller controller(epoch,maximum);
 auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(lifetime_seconds);
 bool failed=false;
 while(controller.life()==Life::live&&!interrupted){
  auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();if(remaining<=0)break;
  pollfd ready{server,POLLIN,0};int result=::poll(&ready,1,int(remaining));if(result<0&&errno==EINTR)continue;if(result<=0)break;
  int client=::accept4(server,nullptr,nullptr,SOCK_CLOEXEC);if(client<0){failed=true;break;}
  pollfd incoming{client,POLLIN,0};result=::poll(&incoming,1,1000);
  if(result>0){std::array<char,512>buffer{};auto n=::recv(client,buffer.data(),buffer.size(),MSG_TRUNC);std::string output="E "+std::to_string(unsigned(Error::invalid));
   if(n>0&&size_t(n)<buffer.size()){
    std::string line(buffer.data(),size_t(n));bool query=line.starts_with("Q ");bool send=line.starts_with("S ");
    if(query||send){auto request=parse(line.substr(2));if(auto r=std::get_if<Request>(&request)){auto lookup=query?controller.query(*r):controller.dispatch(*r,backend,journal);output=render(lookup);if(auto e=std::get_if<Error>(&lookup);e&&*e==Error::journal){failed=true;}}
    }
   }
   ::send(client,output.data(),output.size(),MSG_NOSIGNAL);
  }
  ::close(client);if(failed)break;
 }
 if(controller.life()!=Life::closed){controller.context_lost();if(!journal.record("L "+std::to_string(epoch)+" context_lost_no_checkpoint_reload"))failed=true;}
 else if(!journal.record("C "+std::to_string(epoch)+" exact_all_streams_close"))failed=true;
 ::close(server);::unlink(path.c_str());return controller.life()==Life::closed&&!failed?0:4;
}
}
