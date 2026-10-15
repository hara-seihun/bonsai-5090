#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
struct E{uint32_t bits;int target;};int main(){FILE*f=fopen("native-pair-image.bin","rb");if(!f)return 3;std::vector<E>v(729);if(fread(v.data(),8,729,f)!=729)return 3;fclose(f);std::sort(v.begin(),v.end(),[](E a,E b){return a.bits<b.bits;});v.erase(std::unique(v.begin(),v.end(),[](E a,E b){return a.bits==b.bits;}),v.end());auto start=std::chrono::steady_clock::now();uint32_t rng=0x83170f2dU;for(int log=11;log<=12;log++){std::vector<uint32_t>used(1<<log);for(uint32_t trial=1;trial<=500000;trial++){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;uint32_t mult=rng|1;bool ok=true;for(E e:v){uint32_t key=uint32_t((e.bits>>13)*mult)>>(32-log);if(used[key]==trial){ok=false;break;}used[key]=trial;}if(ok){printf("{\"state\":\"found\",\"keys\":%zu,\"log2_slots\":%d,\"multiplier\":%u,\"trials\":%u,\"table_bytes\":%d}\n",v.size(),log,mult,trial,(1<<log)*4);return 0;}if(trial%1024==0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>1){puts("{\"state\":\"bounded_search_elapsed\"}");return 0;}}}puts("{\"state\":\"tested_family_no_match\"}");}
