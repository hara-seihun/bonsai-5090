#pragma once
#include "gguf.hpp"
namespace halo6000 {
Result<std::string>cache_path(const std::string&source,const std::string&directory);
Result<HostImage>load_cached(const std::string&source,const std::string&directory,bool build_if_missing=true,bool verify_payload=true);
Status verify_image(const Gguf&source,const HostImage&image);
} // namespace halo6000
