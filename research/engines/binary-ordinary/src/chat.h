#pragma once
#include "nlohmann/json.hpp"
#include <string>
#include <vector>

namespace h5 {
inline bool chat_prompt(const nlohmann::ordered_json& msgs, bool think, const std::string& marker,
                        std::string& prompt, std::vector<std::string>& images, std::string& error) {
  if (!msgs.is_array()) { error = "messages must be an array"; return false; }
  prompt.clear(); images.clear();
  for (const auto& m : msgs) {
    if (!m.is_object() || !m.contains("content")) { error = "message requires content"; return false; }
    std::string role = m.value("role", "user"), content;
    const auto& c = m["content"];
    auto text = [&](const std::string& s) {
      if (s.find(marker) != std::string::npos) { error = "reserved media marker in text"; return false; }
      content += s; return true;
    };
    if (c.is_string()) { if (!text(c.get<std::string>())) return false; }
    else if (c.is_array()) {
      for (const auto& p : c) {
        if (!p.is_object()) { error = "invalid content part"; return false; }
        std::string type = p.value("type", "");
        if (type == "text") { if (!text(p.at("text").get<std::string>())) return false; }
        else if (type == "image_url") {
          const auto& image = p.at("image_url");
          const std::string url = image.is_string() ? image.get<std::string>() : image.at("url").get<std::string>();
          if (url.empty() || images.size() >= 8) { error = "provide 1 to 8 nonempty image URLs"; return false; }
          images.push_back(url); content += marker;
        } else { error = "unsupported content part: " + type; return false; }
      }
    } else if (!c.is_null()) { error = "content must be text or a content array"; return false; }
    if (role == "developer") role = "system";
    prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
  }
  prompt += "<|im_start|>assistant\n";
  prompt += think ? "<think>\n" : "<think>\n\n</think>\n\n";
  return true;
}
}
