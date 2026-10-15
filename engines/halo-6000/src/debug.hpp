#pragma once
#include "decode.hpp"
#include <string>
namespace halo6000 {
Status dump_forward(const DeviceModel&model,DecodeWorkspace workspace,const std::string&model_path,const std::string&tokens_file,const std::string&output_dir,int steps,int blocks);
Status dump_layer0(const DeviceModel&model,DecodeWorkspace workspace,const std::string&model_path,const std::string&tokens_file,const std::string&output_dir,int steps,int blocks);
} // namespace halo6000
