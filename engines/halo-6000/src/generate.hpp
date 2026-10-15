#pragma once
#include "decode.hpp"
#include "server.hpp"
namespace halo6000 {
Result<Completion> generate_native(DeviceModel&model,DecodeWorkspace workspace,const Tokenizer&tokenizer,const PreparedChat&prompt,int max_tokens,int grid=188);
} // namespace halo6000
