#include "chat.h"
#include <cassert>
int main() {
  using json = nlohmann::ordered_json;
  std::string prompt, error; std::vector<std::string> images;
  auto msgs = json::array({{{"role", "user"}, {"content", json::array({
    {{"type", "text"}, {"text", "before"}}, {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,YQ=="}}}},
    {{"type", "text"}, {"text", "between"}}, {{"type", "image_url"}, {"image_url", {{"url", "http://example/image.png"}}}},
    {{"type", "text"}, {"text", "after"}}})}}});
  assert(h5::chat_prompt(msgs, false, "<__media__>", prompt, images, error));
  assert(images.size() == 2 && images[1] == "http://example/image.png");
  assert(prompt.find("before<__media__>between<__media__>after") != std::string::npos);
  assert(prompt.find("</think>") != std::string::npos);
  msgs[0]["content"] = "<__media__>";
  assert(!h5::chat_prompt(msgs, false, "<__media__>", prompt, images, error));
  msgs[0]["content"] = json::array({{{"type", "input_audio"}}});
  assert(!h5::chat_prompt(msgs, false, "<__media__>", prompt, images, error));
  msgs = json::array({{{"role", "developer"}, {"content", "hello"}}});
  assert(h5::chat_prompt(msgs, true, "<__media__>", prompt, images, error));
  assert(images.empty() && prompt.find("<|im_start|>system\nhello") == 0);
}
