#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
struct State {uint32_t bits;int target;std::array<int,4> path;};
float quant(float x,int mode){if(!mode)return x;uint32_t u=std::bit_cast<uint32_t>(x);u+=(mode==1?4096:4095+((u>>13)&1));return std::bit_cast<float>(u&0xffffe000u);}
int main(){auto start=std::chrono::steady_clock::now();int ps[5]={2,3,5,7,11};
for(int w=3;w<=5;w++)for(int q=5;q<=8;q++)for(int mode=0;mode<3;mode++){
int count=1;for(int j=0;j<w;j++)count*=3;std::vector<std::array<int,5>> digits(count);std::vector<float> vals(count);double largest=1;for(int j=0;j<w;j++)largest*=std::pow(double(ps[j]),q);int shift=int(std::log2(largest)) /2;
for(int i=0;i<count;i++){int n=i;double v=1;for(int j=w-1;j>=0;j--){int d=n%3;n/=3;digits[i][j]=d-1;v*=std::pow(double(ps[j]),d==2?0:d==0?1:q);}vals[i]=float(std::ldexp(v,-shift));}
std::vector<int> products(count*count);for(int a=0;a<count;a++)for(int b=0;b<count;b++){int t=0;for(int j=0;j<w;j++)t=3*t+digits[a][j]*digits[b][j]+1;products[a*count+b]=t;}
std::vector<State> states;for(int i=0;i<count;i++)states.push_back({std::bit_cast<uint32_t>(vals[i]),i,{i,-1,-1,-1}});bool collision=false,limit=false;int stage=1;std::array<State,2> witness;
for(stage=2;stage<=4;stage++){
if(states.size()*count>12000000){limit=true;break;}
std::vector<State> next;next.reserve(states.size()*count);for(auto &s:states)for(int i=0;i<count;i++){float a=quant(std::bit_cast<float>(s.bits),mode),b=quant(vals[i],mode);float v=a*b;auto p=s.path;p[stage-1]=i;next.push_back({std::bit_cast<uint32_t>(v),products[s.target*count+i],p});}
std::sort(next.begin(),next.end(),[](const State&a,const State&b){return a.bits<b.bits||(a.bits==b.bits&&a.target<b.target);});
for(size_t i=1;i<next.size();i++){if(next[i].bits==next[i-1].bits&&next[i].target!=next[i-1].target){witness={next[i-1],next[i]};collision=true;break;}}if(collision)break;next.erase(std::unique(next.begin(),next.end(),[](const State&a,const State&b){return a.bits==b.bits;}),next.end());states=std::move(next);
}
std::cout<<"{\"width\":"<<w<<",\"q\":"<<q<<",\"mode\":"<<mode<<",\"shift\":"<<shift<<",\"status\":\""<<(collision?"collision":limit?"resource_bound":"sat")<<"\",\"stage\":"<<(stage>4?4:stage)<<",\"prefix_states\":"<<states.size();
if(collision){std::cout<<",\"bits\":"<<witness[0].bits<<",\"paths\":[";for(int a=0;a<2;a++){if(a)std::cout<<",";std::cout<<"[";for(int j=0;j<4;j++){if(j)std::cout<<",";std::cout<<(witness[a].path[j]<0?count-1:witness[a].path[j]);}std::cout<<"]";}std::cout<<"],\"targets\":["<<witness[0].target<<","<<witness[1].target<<"]";}
else if(!limit){std::string name="w"+std::to_string(w)+"-q"+std::to_string(q)+"-m"+std::to_string(mode)+".bin";std::ofstream f(name,std::ios::binary);for(auto&s:states){f.write((char*)&s.bits,4);f.write((char*)&s.target,4);}std::cout<<",\"decoder_file\":\""<<name<<"\"";}
std::cout<<"}"<<std::endl;if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>48)return 0;
}}
