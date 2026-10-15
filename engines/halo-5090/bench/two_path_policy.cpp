#include "two_path_policy.h"
#include <cassert>
#include <cstdio>
int main() {
  int in[8] = {10,20,31,40, 10,20,30,40};
  int out[8] = {20,30,9,8, 20,30,40,50};
  auto c=h5::choose_two_path(in,out,4,1);
  assert(c.error==h5::PathError::None&&c.path==1&&c.matched==3);
  out[4]=99;out[5]=99;
  c=h5::choose_two_path(in,out,4,1);assert(c.path==1&&c.matched==3);
  assert(h5::selected_target(out,4,1,c,0)==20&&h5::selected_target(out,4,1,c,1)==30&&h5::selected_target(out,4,1,c,2)==40);
  out[1]=31;out[2]=40;
  c=h5::choose_two_path(in,out,4,1);assert(c.path==0&&c.matched==3);
  out[0]=99;out[4]=20;
  c=h5::choose_two_path(in,out,4,1);assert(c.path==0&&c.matched==0); // Duplicate cannot rescue an invalid canonical prefix.
  in[5]=21;assert(h5::choose_two_path(in,out,4,1).error==h5::PathError::SharedInput);
  in[5]=20;in[6]=31;assert(h5::choose_two_path(in,out,4,1).error==h5::PathError::DuplicateBranch);
  assert(h5::choose_two_path(in,out,4,3).error==h5::PathError::Geometry);
  puts("two-path canonical exact-tree policy: pass");
}
