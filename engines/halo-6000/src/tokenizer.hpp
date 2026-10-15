#pragma once
#include "format.hpp"
#include <memory>
#include <string>
#include <vector>

namespace halo6000 {
using Token = int32_t;
struct ChatMessage {std::string role,content;};
struct RenderedChat {std::string prompt,generation_prompt;};
struct PreparedChat {RenderedChat rendered;std::vector<Token> tokens;};

// CPU vocabulary/template helper only. No llama context, inference or sampling.
// Text messages only; tools, multimodal content and continuation are not exposed.
class Tokenizer {
 struct Impl;
 std::unique_ptr<Impl> impl_;
 explicit Tokenizer(std::unique_ptr<Impl>);
public:
 ~Tokenizer();
 Tokenizer(const Tokenizer&)=delete;
 Tokenizer&operator=(const Tokenizer&)=delete;
 static Result<std::unique_ptr<Tokenizer>> load(const std::string&gguf);
 // Always add_special=false (no implicit BOS/EOS). parse_special is explicit.
 Result<std::vector<Token>> tokenize(const std::string&text,bool parse_special=true)const;
 // Raw text, no thinking/tool stripping; special token spellings preserved by default.
 Result<std::string> detokenize(const std::vector<Token>&tokens,bool render_special=true)const;
 Result<std::string> piece(Token token,bool render_special=true)const;
 // Embedded GGUF Jinja; enable_thinking=false, reasoning none, pure content.
 Result<RenderedChat> render_chat(const std::vector<ChatMessage>&messages,bool add_generation_prompt=true)const;
 Result<PreparedChat> prepare_chat(const std::vector<ChatMessage>&messages,bool add_generation_prompt=true)const;
 bool is_eog(Token token)const;
 const std::vector<Token>&eog_tokens()const;
 const std::string&template_source()const;
 int32_t vocab_size()const;
};
} // namespace halo6000
