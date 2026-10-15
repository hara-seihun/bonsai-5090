#pragma once
#include <cstdlib>
#include <cstring>
namespace h5 {
enum class FullKScheduleError { None, InvalidTailMask, InvalidOrderedPrep, InvalidGemmFlag, ActiveSchedule };
inline FullKScheduleError fullk_schedule_error(const char* tail, const char* ordered, const char* flag) {
  long mask=0;
  if(tail){char* end=nullptr;mask=std::strtol(tail,&end,10);if(end==tail||!end||*end||mask<0||mask>31)return FullKScheduleError::InvalidTailMask;}
  bool order=false;
  if(ordered){
    if(!std::strcmp(ordered,"off"))order=false;
    else if(!std::strcmp(ordered,"out8")||!std::strcmp(ordered,"out")||!std::strcmp(ordered,"down")||!std::strcmp(ordered,"both"))order=true;
    else return FullKScheduleError::InvalidOrderedPrep;
  }
  bool enabled=false;
  if(flag){if(!std::strcmp(flag,"0"))enabled=false;else if(!std::strcmp(flag,"1"))enabled=true;else return FullKScheduleError::InvalidGemmFlag;}
  return mask||order||enabled ? FullKScheduleError::ActiveSchedule : FullKScheduleError::None;
}
inline const char* fullk_schedule_message(FullKScheduleError e) {
  switch(e){
    case FullKScheduleError::None:return nullptr;
    case FullKScheduleError::InvalidTailMask:return "H5_TAIL_PREP must be a mask 0..31";
    case FullKScheduleError::InvalidOrderedPrep:return "H5_ORDERED_PREP requires off|out8|out|down|both";
    case FullKScheduleError::InvalidGemmFlag:return "H5_GEMM_FLAG must be 0 or 1";
    case FullKScheduleError::ActiveSchedule:return "H5_GU_COOP excludes active tail/ordered/flag schedules";
  }
  std::abort();
}
}
