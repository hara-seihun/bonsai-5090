#include "decode.hpp"
#include <chrono>
#include <cstdio>
using namespace halo6000;
static bool ok(Status s){if(s.ok())return true;fprintf(stderr,"%s\n",s.error().message.c_str());return false;}
int main(int argc,char**argv){
 if(argc!=3){fprintf(stderr,"engine-timing MODEL CACHE_DIR\n");return 2;}
 auto image=load_cached(argv[1],argv[2]);if(!image.ok()){fprintf(stderr,"%s\n",image.error().message.c_str());return 2;}
 auto made=DeviceModel::create(image.value(),1,1024);if(!made.ok()){fprintf(stderr,"%s\n",made.error().message.c_str());return 2;}auto m=std::move(made.value());
 auto workspace=Workspace::create(1024);if(!workspace.ok())return 2;auto w=std::move(workspace.value());
 auto token=[](int p){return 100+(p*31+p*p*7)%20000;};
 cudaEvent_t begin,end;if(!ok(cuda_status(cudaEventCreate(&begin),"begin"))||!ok(cuda_status(cudaEventCreate(&end),"end")))return 2;
 for(int lanes:{32,16,8,16,32}){
  m->matvec_lanes=lanes;if(!ok(m->reset()))return 2;
  for(int p=0;p<40;p++)if(!ok(run_token(*m,w->view,token(p),p,0,188)))return 2;
  if(!ok(cuda_status(cudaDeviceSynchronize(),"warm")))return 2;
  auto wall=std::chrono::steady_clock::now();if(!ok(cuda_status(cudaEventRecord(begin),"timing begin")))return 2;
  for(int p=40;p<168;p++)if(!ok(run_token(*m,w->view,token(p),p,0,188)))return 2;
  if(!ok(cuda_status(cudaEventRecord(end),"timing end"))||!ok(cuda_status(cudaEventSynchronize(end),"timing complete")))return 2;
  float ms=0;if(!ok(cuda_status(cudaEventElapsedTime(&ms,begin,end),"timing ms")))return 2;
  double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count();
  printf("{\"kind\":\"bench\",\"row_lanes\":%d,\"grid\":188,\"warm\":40,\"steps\":128,\"gpu_ms\":%.6f,\"wall_ms\":%.6f,\"tokens_per_second\":%.6f,\"weights_bytes\":%zu,\"prefix_route\":\"all_serial_teacher_forcing\"}\n",lanes,ms,wall_ms,128000./ms,m->weight_bytes);fflush(stdout);
 }
 cudaEventDestroy(begin);cudaEventDestroy(end);
 auto profiling=Workspace::create(1024,true);if(!profiling.ok())return 2;auto p=std::move(profiling.value());
 for(int lanes:{32,16,8}){
  m->matvec_lanes=lanes;if(!ok(m->reset()))return 2;
  for(int pos=0;pos<=40;pos++)if(!ok(run_token(*m,p->view,token(pos),pos,0,188)))return 2;
  uint64_t times[646];if(!ok(cuda_status(cudaMemcpy(times,p->view.timestamps,sizeof times,cudaMemcpyDeviceToHost),"profile read")))return 2;
  printf("{\"kind\":\"profile\",\"row_lanes\":%d,\"grid\":188,\"position\":40,\"timestamps\":[",lanes);for(int i=0;i<646;i++)printf("%s%llu",i?",":"",(unsigned long long)times[i]);puts("]}");fflush(stdout);
 }
 return 0;
}
