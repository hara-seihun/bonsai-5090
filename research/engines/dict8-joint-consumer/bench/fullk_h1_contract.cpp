#include "../src/fullk_profile.h"
#include <cassert>
#include <cstdio>
#include <vector>
int main(){
  using h5::FullKScheduleError;using h5::fullk_schedule_error;
  assert(fullk_schedule_error(nullptr,nullptr,nullptr)==FullKScheduleError::None);
  assert(fullk_schedule_error("0","off","0")==FullKScheduleError::None);
  assert(fullk_schedule_error("00","off","0")==FullKScheduleError::None);
  assert(fullk_schedule_error("1","off","0")==FullKScheduleError::ActiveSchedule);
  assert(fullk_schedule_error("0","out","0")==FullKScheduleError::ActiveSchedule);
  assert(fullk_schedule_error("0","off","1")==FullKScheduleError::ActiveSchedule);
  assert(fullk_schedule_error("32","off","0")==FullKScheduleError::InvalidTailMask);
  assert(fullk_schedule_error("x","off","0")==FullKScheduleError::InvalidTailMask);
  assert(fullk_schedule_error("0","0","0")==FullKScheduleError::InvalidOrderedPrep);
  assert(fullk_schedule_error("0","off","off")==FullKScheduleError::InvalidGemmFlag);
  constexpr int rows=64,features=17408;
  std::vector<unsigned char> writes(rows*features),pairs(rows*features/2);
  for(int bx=0;bx<2;bx++)for(int by=0;by<68;by++)for(int warp=0;warp<8;warp++)
    for(int lane=0;lane<32;lane++)for(int n=0;n<4;n++)for(int j=0;j<2;j++)for(int mm=0;mm<2;mm++)for(int r=0;r<2;r++){
      int gid=lane>>2,tq=lane&3,row=bx*32+n*8+tq*2+j;
      int fg=(by*16+warp*2)/2,feature=fg*32+gid+mm*16+r*8;
      int peer=lane^4,pf=fg*32+(peer>>2)+mm*16+r*8,pr=bx*32+n*8+(peer&3)*2+j;
      assert(pf==(feature^1)&&pr==row);
      assert(++writes[row*features+feature]==1);
      if(!(gid&1))assert(++pairs[(row*features+feature)/2]==1);
    }
  for(auto n:writes)assert(n==1);for(auto n:pairs)assert(n==1);
  for(int row=0;row<64;row++)for(int chunk=0;chunk<17;chunk++)for(int warp=0;warp<4;warp++)for(int lane=0;lane<32;lane++)for(int k=0;k<4;k++){
    int feature=chunk*1024+warp*256+lane*8+2*k;
    int pair=row*(features/2)+feature/2;assert(pairs[pair]==1);
  }
  std::printf("schedule10pass; cfg13: %zu feature coordinates, %zu carrier pairs; xor4 preserves row/flips feature-bit0; complete consumer coverage\n",writes.size(),pairs.size());
}
