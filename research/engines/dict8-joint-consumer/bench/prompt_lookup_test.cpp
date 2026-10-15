#include "prompt_lookup.h"
#include <cassert>
#include <cstdio>
int main(){
  h5::lookup::Index ix;
  int p[]={1,2,3,4,5,6,7,8,9,10,11,12,1,2,3};
  ix.append(p,15);
  auto h=ix.continuation(8); assert(h.length==8 && h.tokens[0]==4 && h.tokens[7]==11);
  int q[]={4,5}; ix.append(q,2); h=ix.continuation(20); assert(h.length==12 && h.tokens[0]==6); // 6..12 then up to len
  h5::lookup::Index e; int z[]={5,6,7}; e.append(z,3); assert(e.continuation(4).length==0);
  printf("lookup ok %d\n", h.length);
}
