#pragma once
#include "format.hpp"
#include <cstring>
#include <limits>
namespace halo6000 {
inline constexpr size_t Q4Raw32Bytes=18,Q4Grouped128Bytes=72;
inline Status repack_q4_0_grouped128(const uint8_t*source,size_t blocks32,uint8_t*dest,size_t bytes){
 if(blocks32%4||blocks32>std::numeric_limits<size_t>::max()/18||bytes!=blocks32*18||
    (blocks32&&(!source||!dest||source==dest)))return Error{ErrorCode::Invalid,"Q4_0 grouped128 repack extent/buffer"};
 for(size_t g=0;g<blocks32/4;g++)for(int c=0;c<4;c++){
  std::memcpy(dest+g*72+c*16,source+(g*4+c)*18+2,16);
  std::memcpy(dest+g*72+64+c*2,source+(g*4+c)*18,2);
 }
 return success();
}
} // namespace halo6000
