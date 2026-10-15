#pragma once
#include "tokenizer.hpp"
#include <functional>
#include <memory>

namespace halo6000 {
struct Completion {
 // All sampled IDs, including final EOG if sampled. EOG counts toward usage,
 // but never content. Return only after GPU completion; no queue time in ms.
 std::vector<Token> tokens;
 int32_t completion_tokens=0;
 double prompt_ms=0,predicted_ms=0;
};
using Generator=std::function<Result<Completion>(const PreparedChat&,int32_t max_tokens)>;
struct ServerConfig {
 std::string host="127.0.0.1",model="evaluated-model";
 int port=8080,context_tokens=4096,max_tokens=1024;
 size_t max_request_bytes=1024*1024;
};
class HttpServer {
 struct Impl;
 std::unique_ptr<Impl> impl_;
 explicit HttpServer(std::unique_ptr<Impl>);
public:
 ~HttpServer();
 HttpServer(const HttpServer&)=delete;
 HttpServer&operator=(const HttpServer&)=delete;
 // Generator must reset native state each call. create probes actual generation
 // before any socket bind, failing on unloaded/broken execution. Not a GPU batcher.
 static Result<std::unique_ptr<HttpServer>> create(ServerConfig,std::shared_ptr<Tokenizer>,Generator);
 // Port 0 chooses an ephemeral port. bind() follows the successful startup probe.
 Result<int> bind();
 Status listen(); // blocking foreground; call stop() then join caller's thread.
 void stop();
};
} // namespace halo6000
