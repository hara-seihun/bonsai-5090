#pragma once
#include "decode.hpp"
#include <string>

namespace halo6000 {
Status dump_intrinsic(DeviceModel & model, DecodeWorkspace workspace,
                      const Provenance & provenance, const std::string & spec_path,
                      const std::string & output_dir, int grid);
}
