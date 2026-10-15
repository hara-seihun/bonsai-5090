#include "native.cu"
#include "../transport.hpp"
namespace live17 {
struct StreamBackend {
 native14::Receiver receiver;Journal&journal;std::vector<float>x;unsigned next_layer=0;bool active_candidate=false,closed=false;std::optional<Request>domain;std::optional<binding15::Fault> configured_fault;
 explicit StreamBackend(Journal&j,unsigned stream):journal(j){binding15::active=&receiver.effects;receiver.stream=stream;}
 Reply response(unsigned code,bool done=false){auto*p=receiver.effects.protocol();return {code,done,p&&p->head()?p->head()->generation():0,receiver.effects.events().size()};}
 Reply error(native14::Error e){return response(100+unsigned(e));}
 Reply result(bonsai14::Result<bool> r){if(auto e=std::get_if<bonsai14::Error>(&r))return response(200+unsigned(*e));return response(0,std::get<bool>(r));}
 Reply reconcile(const Request&r){
  auto&b=receiver.effects;auto*p=b.protocol();std::span<const bonsai14::HandleView>handles;
  uint64_t generation=b.reset_generation(),boundary=b.reset_boundary();
  switch(r.scope){
   case 0:handles=b.reset_handles();if(p)return response(300);break;
   case 1:if(!p||!b.input())return response(300);handles=p->candidate_handles();generation=b.input()->generation;boundary=b.input()->boundary;break;
   case 2:if(!p||!p->retained_old())return response(300);handles=p->old_handles();generation=(*p->retained_old())->generation();boundary=(*p->retained_old())->boundary();break;
   case 3:if(!p||!p->head())return response(300);handles=p->head_handles();generation=p->head()->generation();boundary=p->head()->boundary();break;
   default:return response(300);
  }
  if(r.generation!=generation||r.boundary!=boundary||r.ticket>=handles.size())return response(300);
  auto h=handles[r.ticket];if(!h.native||*h.native!=r.handle)return response(300);
  auto*original=binding15::evidence_record(r.handle);if(!original||original->stream!=r.stream||original->domain!=binding15::current_domain||original->owner.layer!=r.layer||original->owner.generation!=r.generation||original->owner.boundary!=r.boundary)return response(300);
  auto evidence=binding15::known(r.handle);if(!evidence||*evidence==binding15::Evidence::unknown)return response(301);
  bonsai14::Status status=bonsai14::Error::order;
  if(h.state==bonsai14::HandleState::uncertain_acquisition&&*evidence==binding15::Evidence::live){status=r.scope==0?b.resolve({r.ticket,generation},bonsai14::Acquired{r.handle}):r.scope==1?p->resolve_handle({r.ticket,generation},bonsai14::Acquired{r.handle}):bonsai14::Status{bonsai14::Error::order};}
  else if(h.state==bonsai14::HandleState::uncertain_release||h.state==bonsai14::HandleState::failed_release){auto outcome=*evidence==binding15::Evidence::released?bonsai14::Outcome::success:bonsai14::Outcome::failure;
   if(r.scope==1){for(auto&v:p->validation_records())if(v.ticket.index==r.ticket&&v.ticket.generation==r.generation&&v.native==r.handle){bonsai14::ValidationHandle exact{v.ticket,v.owner,v.range,h.kind,r.handle};status=b.reconcile_validation_release(exact,outcome);return binding15::ok(status)?response(0,true):response(302);}}
   switch(r.scope){case 0:status=b.reconcile_reset_release(r.ticket,outcome);break;case 1:status=p->reconcile_abort_release(r.ticket,outcome);break;case 2:status=p->reconcile_retirement_release(r.ticket,outcome);break;case 3:status=p->reconcile_head_release(r.ticket,outcome);break;}
  }
  return binding15::ok(status)?response(0,true):response(302);
 }
 Reply execute(const Request&r) try {
  using C=Command;binding15::active=&receiver.effects;
  if(closed)return response(300);if(r.command==C::reconcile)return reconcile(r);
  if(r.ticket||r.handle||r.scope)return response(300);
  if(r.command==C::open||r.command==C::reset_layer||r.command==C::bootstrap){
   if(r.generation!=1||r.boundary!=0||r.accepted||r.fresh!=1||r.keep!=1||r.replay!=1)return response(300);
   if(r.command==C::open){if(r.layer||receiver.opened)return response(300);if(!binding15::before(binding15::Effect::attribute)||binding15::finish(cudaSetDevice(0))!=cudaSuccess)return error(native14::Error::launch);cudaDeviceProp props{};if(!binding15::before(binding15::Effect::attribute)||binding15::finish(cudaGetDeviceProperties(&props,0))!=cudaSuccess)return error(native14::Error::launch);if(props.major!=12||props.minor!=0||std::string(props.name).find("RTX 5090")==std::string::npos)return error(native14::Error::input);return receiver.initialize_open()?response(0,true):error(native14::Error::input);}
   if(r.command==C::reset_layer)return receiver.initialize_layer(r.layer)?response(0,true):error(native14::Error::allocation);
   if(r.layer)return response(300);return receiver.initialize_seal()?response(0,true):error(native14::Error::publication);
  }
  if(r.command==C::close&&!receiver.effects.protocol()){
   if(r.generation!=receiver.effects.reset_generation()||r.boundary!=receiver.effects.reset_boundary()||r.layer)return response(300);
   if(!receiver.clone_origin&&(r.accepted||r.fresh!=1||r.keep!=1||r.replay!=1))return response(308);
   auto out=receiver.effects.reset_close_step(binding15::cleanup_fence(),binding15::release);if(auto done=std::get_if<bool>(&out);done&&*done)closed=true;return result(out);
  }
  auto*p=receiver.effects.protocol();if(!p||!p->head())return response(300);
  if(r.command==C::begin){
   if((r.fresh!=1&&r.fresh!=8)||active_candidate||r.layer||r.generation!=p->head()->generation()+1||r.boundary!=p->head()->boundary()+1||r.boundary>9)return response(300);
   auto&old=receiver.generations[receiver.head];if(r.accepted&&(!old.keep||r.accepted>old.rows||r.replay!=old.replay))return response(306);if(old.committed+r.accepted+r.fresh>32)return response(307);
   receiver.accepted=r.accepted;receiver.rows=r.fresh;receiver.keep=r.keep;receiver.replay=r.replay;domain=r;x.clear();for(unsigned row=0;row<r.fresh;row++){std::vector<float>one;if(!native14::take(receiver.model.embedding(1001+receiver.stream*96+old.committed+r.accepted+row),one))return error(native14::Error::input);x.insert(x.end(),one.begin(),one.end());}
   auto e=receiver.begin(unsigned(r.boundary)-1,x);if(e!=native14::Error::ok)return error(e);active_candidate=true;next_layer=0;return response(0,true);
  }
  bool candidate=r.command==C::acquire||r.command==C::release||r.command==C::publish||r.command==C::abort;
  if(candidate&&(!domain||domain->accepted!=r.accepted||domain->fresh!=r.fresh||domain->keep!=r.keep||domain->replay!=r.replay))return response(308);
  if(candidate){auto*i=receiver.effects.input();if(!i||!active_candidate||r.generation!=i->generation||r.boundary!=i->boundary)return response(300);}
  else if(r.generation!=p->head()->generation()||r.boundary!=p->head()->boundary())return response(300);
  if(!candidate){auto&head=receiver.generations[receiver.head];if(!receiver.head&&(r.accepted||r.fresh!=1||r.keep!=1||r.replay!=1))return response(308);if(receiver.head&&(r.accepted!=receiver.generations[receiver.head].layers[0]->accepted||r.fresh!=head.rows||r.keep!=head.keep||r.replay!=head.replay))return response(308);}
  unsigned position=unsigned(r.boundary)-1;
  switch(r.command){
   case C::acquire:{if(r.layer!=next_layer)return response(300);auto acquired=receiver.acquire(position,r.layer,1001+receiver.stream*96+receiver.generations[receiver.head].committed+receiver.accepted,x);if(auto e=std::get_if<native14::Error>(&acquired))return error(*e);x=std::get<native14::Lease*>(acquired)->output;return response(0,true);}
   case C::release:{if(r.layer!=next_layer)return response(300);auto e=receiver.release(position,r.layer);if(auto*failure=std::get_if<native14::Error>(&e))return error(*failure);bool done=std::get<bool>(e);if(done)++next_layer;return response(0,done);}
   case C::publish:{if(r.layer||next_layer!=64)return response(300);auto e=receiver.publish(position);if(e!=native14::Error::ok)return error(e);active_candidate=false;x.clear();return response(0,true);}
   case C::retire:{if(r.layer)return response(300);auto result=receiver.effects.retire_step(binding15::release);if(auto done=std::get_if<bool>(&result);done&&*done){auto&g=receiver.generations[receiver.head-1];for(auto&v:g.layers)v.reset();g.image.reset();g.published=false;}return this->result(result);}
   case C::abort:{if(r.layer)return response(300);auto result=receiver.effects.abort_step(binding15::cleanup_fence(),binding15::release);if(auto done=std::get_if<bool>(&result);done&&*done){for(auto&v:receiver.generations[position+1].layers)v.reset();active_candidate=false;x.clear();}return this->result(result);}
   case C::close:{if(r.layer)return response(300);auto result=receiver.effects.close_step(binding15::cleanup_fence(),binding15::release);if(auto done=std::get_if<bool>(&result);done&&*done)closed=true;if(auto done=std::get_if<bool>(&result);done&&*done){for(auto&g:receiver.generations){for(auto&v:g.layers)v.reset();g.image.reset();g.published=false;}}return this->result(result);}
   case C::open:case C::reset_layer:case C::bootstrap:case C::begin:case C::reconcile:case C::clone_layer:case C::clone_seal:return response(300);
  }
  return response(300);
 }catch(const std::bad_alloc&){receiver.effects.reject(bonsai14::Error::host_allocation);return response(304);}catch(const std::length_error&){receiver.effects.reject(bonsai14::Error::capacity);return response(305);}
};
struct NativeBackend {
 Journal&journal;unsigned batch;std::vector<std::unique_ptr<StreamBackend>>streams;
 NativeBackend(Journal&j,unsigned B):journal(j),batch(B){binding15::evidence.reserve(size_t(B)*524288);streams.reserve(B);for(unsigned stream=0;stream<B;stream++)streams.push_back(std::make_unique<StreamBackend>(j,stream));}
 bool all_closed()const{for(auto&s:streams)if(!s->closed)return false;return true;}
 Reply execute(const Request&r) try {if(r.batch!=batch||r.stream>=batch)return {300,false,0,0};binding15::current_stream=r.stream;binding15::current_domain={r.accepted,r.fresh,r.keep,r.replay,unsigned(r.boundary?r.boundary-1:0)%2};if(r.command==Command::clone_layer||r.command==Command::clone_seal){
   auto&destination=*streams[r.stream];binding15::active=&destination.receiver.effects;
   if(r.handle>=batch||r.handle==r.stream||r.ticket||r.scope||destination.closed)return destination.response(309);
   auto&source=*streams[r.handle];auto*p=source.receiver.effects.protocol();if(source.closed||source.active_candidate||!p||!p->head()||p->retained_old()||r.generation!=p->head()->generation()||r.boundary!=p->head()->boundary())return destination.response(309);
   auto&h=source.receiver.generations[source.receiver.head];if(source.receiver.head){auto&l=*h.layers[0];if(r.accepted!=l.accepted||r.fresh!=h.rows||r.keep!=h.keep||r.replay!=h.replay)return destination.response(308);}else if(r.accepted||r.fresh!=1||r.keep!=1||r.replay!=1)return destination.response(308);
   if(r.command==Command::clone_layer)return destination.receiver.initialize_clone_layer(source.receiver,r.layer)?destination.response(0,true):destination.response(309);
   if(r.layer||!destination.receiver.clone_origin||*destination.receiver.clone_origin!=std::array<uint64_t,3>{r.handle,r.generation,r.boundary})return destination.response(309);
   return destination.receiver.initialize_seal()?destination.response(0,true):destination.response(309);
  }return streams[r.stream]->execute(r);}catch(const std::bad_alloc&){auto&s=*streams[r.stream];s.receiver.effects.reject(bonsai14::Error::host_allocation);return s.response(304);}catch(const std::length_error&){auto&s=*streams[r.stream];s.receiver.effects.reject(bonsai14::Error::capacity);return s.response(305);}
 bool custody(const Request&r){std::string temp=journal.directory()+"/custody.tmp",path=journal.directory()+"/custody.json";FILE*f=std::fopen(temp.c_str(),"w");if(!f)return false;
  std::fprintf(f,"{\"epoch\":%llu,\"request\":%llu,\"batch\":%u,\"stream\":%u,\"live_context_evidence_only\":true,\"streams\":[",(unsigned long long)r.epoch,(unsigned long long)r.id,batch,r.stream);
  for(unsigned stream=0;stream<batch;stream++){if(stream)std::fputc(',',f);std::fprintf(f,"{\"stream\":%u,\"closed\":%s,\"ledger\":",stream,streams[stream]->closed?"true":"false");streams[stream]->receiver.effects.dump(f);std::fputs(",\"actual_source_families\":[",f);bool comma=false;for(auto&g:streams[stream]->receiver.generations)for(auto&l:g.layers)if(l)for(auto&s:l->families){if(comma)std::fputc(',',f);comma=true;s->dump(f);}std::fputs("]}",f);}
  std::fputs("],\"native_original_evidence\":[",f);for(size_t i=0;i<binding15::evidence_count;i++){auto&e=binding15::evidence[i];if(i)std::fputc(',',f);std::fprintf(f,"{\"handle\":%llu,\"state\":%u,\"stream\":%u,\"accepted\":%u,\"fresh\":%u,\"keep\":%u,\"replay\":%u,\"parity\":%u,\"layer\":%u,\"generation\":%llu,\"boundary\":%llu}",(unsigned long long)e.handle,unsigned(e.state),e.stream,e.domain[0],e.domain[1],e.domain[2],e.domain[3],e.domain[4],e.owner.layer,(unsigned long long)e.owner.generation,(unsigned long long)e.owner.boundary);}std::fputs("]}\n",f);
  bool good=std::fflush(f)==0&&::fsync(fileno(f))==0;if(std::fclose(f))good=false;if(!good||::rename(temp.c_str(),path.c_str()))return false;int fd=::open(journal.directory().c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);if(fd<0)return false;good=::fsync(fd)==0;::close(fd);return good;
 }
};
}
int main(int argc,char**argv){
 if(argc==2&&!std::strcmp(argv[1],"--describe-native22")){std::printf("{\"state\":\"source_ABI_only\",\"GPU_calls\":0,\"row_view_bytes\":%zu,\"borrow_bytes\":%zu,\"borrow_handle_bytes\":%zu,\"FFN_route_bytes\":%zu,\"GDN_route_bytes\":%zu,\"route_union_bytes\":%zu}\n",sizeof(views21::RowView),sizeof(views21::Borrow),sizeof(views21::Borrowed),sizeof(joined12::StridedFfn),sizeof(joined12::StridedGdn),sizeof(joined12::BorrowRoute));return 0;}
 if((argc!=8&&argc!=13)||std::strcmp(argv[1],"--serve-native22")||std::strcmp(argv[argc-1],"--authorized-live-GPU-context")){std::puts("{\"state\":\"authorization_required_or_invalid_arguments\",\"GPU_calls\":0}");return 2;}
 uint64_t epoch=0,maximum=0,lifetime=0,batch=0;auto number=[](char*s,uint64_t&out){auto end=s+strlen(s);auto r=std::from_chars(s,end,out);return r.ec==std::errc{}&&r.ptr==end;};
 if(!number(argv[3],epoch)||!number(argv[4],maximum)||!number(argv[5],lifetime)||!number(argv[6],batch)||!epoch||!maximum||maximum>32768||!lifetime||lifetime>3600||(batch!=1&&batch!=8&&batch!=32)){std::puts("{\"state\":\"invalid_arguments\",\"GPU_calls\":0}");return 2;}
 std::optional<std::pair<unsigned,binding15::Fault>> fault;
 if(argc==13){uint64_t stream=0,event=0;if(std::strcmp(argv[7],"--fault")||!number(argv[8],stream)||stream>=batch||!number(argv[9],event)||event>=524288||(std::strcmp(argv[10],"failure")&&std::strcmp(argv[10],"uncertain"))||(std::strcmp(argv[11],"before")&&std::strcmp(argv[11],"after"))){std::puts("{\"state\":\"invalid_fault_arguments\",\"GPU_calls\":0}");return 2;}fault=std::pair{unsigned(stream),binding15::Fault{size_t(event),!std::strcmp(argv[10],"failure")?bonsai14::Outcome::failure:bonsai14::Outcome::uncertain,!std::strcmp(argv[11],"after")}};}
 try{live17::Journal journal(argv[2]);live17::NativeBackend backend(journal,unsigned(batch));if(fault)backend.streams[fault->first]->receiver.effects.fault(fault->second);return live17::serve(backend,journal,epoch,size_t(maximum),unsigned(lifetime));}catch(const std::bad_alloc&){std::puts("{\"state\":\"host_allocation_failed_before_GPU\",\"GPU_calls\":0}");return 3;}
}
