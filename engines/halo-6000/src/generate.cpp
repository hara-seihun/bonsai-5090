#include "generate.hpp"
namespace halo6000 {
namespace {
struct Events {
 cudaEvent_t begin=nullptr,end=nullptr;
 ~Events(){if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}
 Status create(){auto s=cuda_status(cudaEventCreate(&begin),"generation begin event");if(!s.ok())return s;return cuda_status(cudaEventCreate(&end),"generation end event");}
 Status start(){return cuda_status(cudaEventRecord(begin),"generation phase start");}
 Result<double>finish(){auto s=cuda_status(cudaEventRecord(end),"generation phase end");if(!s.ok())return s.error();s=cuda_status(cudaEventSynchronize(end),"generation phase completion");if(!s.ok())return s.error();float ms=0;s=cuda_status(cudaEventElapsedTime(&ms,begin,end),"generation elapsed");if(!s.ok())return s.error();return double(ms);}
};
}
Result<Completion>generate_native(DeviceModel&m,DecodeWorkspace w,const Tokenizer&tokenizer,const PreparedChat&prompt,int max_tokens,int grid){
 if(prompt.tokens.empty()||max_tokens<1||prompt.tokens.size()+size_t(max_tokens)>size_t(m.state.context))return Error{ErrorCode::Capacity,"prompt+generation exceeds context"};
 auto s=m.reset();if(!s.ok())return s.error();Events events;s=events.create();if(!s.ok())return s.error();s=events.start();if(!s.ok())return s.error();
 for(size_t p=0;p<prompt.tokens.size();p++){s=run_token(m,w,prompt.tokens[p],p,0,grid);if(!s.ok())return s.error();}
 auto prefill=events.finish();if(!prefill.ok())return prefill.error();Completion result;result.prompt_ms=prefill.value();result.tokens.reserve(max_tokens);s=events.start();if(!s.ok())return s.error();
 for(int j=0;j<max_tokens;j++){
  Token token=-1;s=cuda_status(cudaMemcpy(&token,w.token,4,cudaMemcpyDeviceToHost),"native sampled token");if(!s.ok())return s.error();if(token<0||token>=Vocab)return Error{ErrorCode::Invalid,"native argmax did not produce a vocabulary token"};
  result.tokens.push_back(token);if(tokenizer.is_eog(token)||j+1==max_tokens)break;
  s=run_token(m,w,-1,prompt.tokens.size()+j,0,grid);if(!s.ok())return s.error();
 }
 auto decode=events.finish();if(!decode.ok())return decode.error();result.predicted_ms=decode.value();result.completion_tokens=result.tokens.size();return result;
}
} // namespace halo6000
