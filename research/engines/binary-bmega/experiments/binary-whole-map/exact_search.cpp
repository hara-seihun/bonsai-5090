#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>
#include <chrono>
#include <cfenv>
#include <limits>

static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits==24);

using Input = std::array<int,3>;
using Output = std::array<uint32_t,2>;
using Mask = std::array<uint64_t,3>;
struct Direction { Input delta; int left=-1,right=-1; };
struct Carrier { Input twice_coeff; Mask collisions{}; uint64_t survivors=0; int network=-1,left=-1,right=-1; Output observed_left{},observed_right{}; };
struct Network { int g0,u0,g1,u1,d0,d1; };
constexpr int levels[15]={-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12};
int sign(int pattern,int lane) { return (pattern>>lane&1)?1:-1; }
int dot(Input a,Input b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }
float value(uint32_t x) { return std::bit_cast<float>(x); }
bool intersects(Mask a,Mask b) { return (a[0]&b[0])||(a[1]&b[1])||(a[2]&b[2]); }
void print_input(Input x) { printf("[%d,%d,%d]",x[0],x[1],x[2]); }
void print_output(Output y) { printf("{\"bits\":[%u,%u],\"value\":[%.9g,%.9g]}",y[0],y[1],value(y[0]),value(y[1])); }
void print_network(Network n) { printf("[%d,%d,%d,%d,%d,%d]",n.g0,n.u0,n.g1,n.u1,n.d0,n.d1); }

int main() {
  if(std::fesetround(FE_TONEAREST)!=0) {std::fprintf(stderr,"error: cannot establish round-to-nearest-even\n");return 8;}
  const auto begin=std::chrono::steady_clock::now();
  std::array<Input,64> inputs{};
  for(int i=0;i<64;i++) inputs[i]={i%4-2,i/4%4-2,i/16-2};
  std::vector<Direction> directions;
  int direction_id[7][7][7];
  for(auto &a:direction_id) for(auto &b:a) for(int &c:b)c=-1;
  for(int a=-3;a<=3;a++)for(int b=-3;b<=3;b++)for(int c=-3;c<=3;c++) {
    if(a>0||(a==0&&b>0)||(a==0&&b==0&&c>0)) {
      int id=int(directions.size()); directions.push_back({{a,b,c}});
      direction_id[a+3][b+3][c+3]=direction_id[-a+3][-b+3][-c+3]=id;
    }
  }
  std::vector<Carrier> carriers;
  for(int a:levels)for(int b:levels)for(int c:levels) {
    Carrier h{{a,b,c}};
    for(int d=0;d<int(directions.size());d++) if(dot(h.twice_coeff,directions[d].delta)==0) h.collisions[d/64]|=uint64_t(1)<<(d%64);
    carriers.push_back(h);
  }
  float atom[2][8][8][64];
  constexpr float gs[2][2]={{.125f,.5f},{.25f,.125f}};
  constexpr float us[2][2]={{.5f,.25f},{.125f,.5f}};
  for(int f=0;f<2;f++)for(int g=0;g<8;g++)for(int u=0;u<8;u++)for(int i=0;i<64;i++) {
    Input x=inputs[i];
    int gfirst=sign(g,0)*x[0]+sign(g,1)*x[1],glast=sign(g,2)*x[2];
    int ufirst=sign(u,0)*x[0]+sign(u,1)*x[1],ulast=sign(u,2)*x[2];
    float gate=std::fma(float(glast),gs[f][1],std::fma(float(gfirst),gs[f][0],0.f));
    float up=std::fma(float(ulast),us[f][1],std::fma(float(ufirst),us[f][0],0.f));
    atom[f][g][u][i]=(gate/(1.f+std::exp(-gate)))*up;
  }
  uint64_t networks=0,no_carrier=0,total_survivors=0,injective_carriers=0;
  for(auto c:carriers)if(c.collisions==Mask{})injective_carriers++;
  std::map<int,uint64_t> fibres_hist,survivors_hist;
  std::map<uint64_t,std::vector<std::array<Output,64>>> truth_classes;
  int unique_truths=0;
  std::array<Output,64> first_truth{};
  Network first_network{};
  bool nonadditive_emitted=false;
  uint64_t predicted_structured=0,structured_mismatches=0;
  std::array<Output,32> packed_truth{}; std::array<bool,32> packed_reachable{};
  printf("{\"kind\":\"contract\",\"input\":\"signed_A2_three_variables_in_two_K128_groups_with_reachable_A8_anchors\",\"points\":64,\"hidden\":2,\"hidden_zero_padding\":126,\"down_rows\":2,\"gate_scale\":[[0.125,0.5],[0.25,0.125]],\"up_scale\":[[0.5,0.25],[0.125,0.5]],\"down_scale\":[0.0625,0.125],\"carrier_coefficients_twice\":[-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12],\"carriers\":3375,\"semantics\":\"host_FP32_fma_exp_nearbyint_round_nearest_even_exact_bit_keys\"}\n");
  for(int g0=0;g0<8;g0++)for(int u0=0;u0<8;u0++)for(int g1=0;g1<8;g1++)for(int u1=0;u1<8;u1++)for(int d0=0;d0<4;d0++)for(int d1=0;d1<4;d1++) {
    Network n{g0,u0,g1,u1,d0,d1};
    std::array<Output,64> truth{};
    for(int i=0;i<64;i++) {
      float a=atom[0][g0][u0][i],b=atom[1][g1][u1][i];
      float m=std::fmax(std::fabs(a),std::fabs(b));
      int qa=0,qb=0; float sx=0;
      if(m!=0) { float inv=127.f/m;qa=int(std::nearbyint(a*inv));qb=int(std::nearbyint(b*inv));sx=m/127.f; }
      truth[i]={bits(std::fma(float(sign(d0,0)*qa+sign(d0,1)*qb),.0625f*sx,0.f)),bits(std::fma(float(sign(d1,0)*qa+sign(d1,1)*qb),.125f*sx,0.f))};
    }
    if(networks==0) {
      first_truth=truth;first_network=n;
      for(int i=0;i<64;i++) {
        int label=dot({1,1,8},inputs[i])+20;
        if(packed_reachable[label]&&packed_truth[label]!=truth[i])return 4;
        packed_reachable[label]=true;packed_truth[label]=truth[i];
      }
    }
    uint64_t hash=1469598103934665603ull;
    for(auto y:truth)for(uint32_t b:y){hash^=b;hash*=1099511628211ull;}
    auto &bucket=truth_classes[hash];
    bool known=false;for(auto &t:bucket)if(t==truth){known=true;break;}
    if(!known){bucket.push_back(truth);unique_truths++;}
    std::map<Output,int> fibres;for(auto y:truth)fibres[y]++;
    fibres_hist[int(fibres.size())]++;
    Mask bad{}; std::array<int,171> left{},right{};left.fill(-1);right.fill(-1);
    for(int i=0;i<64;i++)for(int j=i+1;j<64;j++)if(truth[i]!=truth[j]) {
      Input delta{inputs[j][0]-inputs[i][0],inputs[j][1]-inputs[i][1],inputs[j][2]-inputs[i][2]};
      int d=direction_id[delta[0]+3][delta[1]+3][delta[2]+3];
      bad[d/64]|=uint64_t(1)<<(d%64);if(left[d]<0){left[d]=i;right[d]=j;}
    }
    if(networks==16) {
      printf("{\"kind\":\"fixed_dense_control\",\"network_id\":16,\"network\":");print_network(n);printf(",\"fibres\":%zu}\n",fibres.size());
      for(int i=0;i<64;i++) {printf("{\"kind\":\"fixed_dense_truth\",\"index\":%d,\"output\":",i);print_output(truth[i]);printf("}\n");}
      for(int id=0;id<int(carriers.size());id++) {
        auto &c=carriers[id];
        if(!intersects(bad,c.collisions))return 6;
        int a=-1,b=-1;
        for(int d=0;d<171;d++)if((bad[d/64]&c.collisions[d/64])>>(d%64)&1){a=left[d];b=right[d];break;}
        if(dot(c.twice_coeff,inputs[a])!=dot(c.twice_coeff,inputs[b])||truth[a]==truth[b])return 7;
        printf("{\"kind\":\"fixed_dense_rejection\",\"carrier_id\":%d,\"left_index\":%d,\"right_index\":%d,\"left_output\":",id,a,b);print_output(truth[a]);printf(",\"right_output\":");print_output(truth[b]);printf("}\n");
      }
    }
    int survivors=0;
    for(auto &c:carriers) {
      if(!intersects(bad,c.collisions)) { c.survivors++;survivors++; }
      else if(c.network<0) {
        c.network=int(networks);
        for(int d=0;d<171;d++)if((bad[d/64]&c.collisions[d/64])>>(d%64)&1){c.left=left[d];c.right=right[d];break;}
        c.observed_left=truth[c.left];c.observed_right=truth[c.right];
        if(dot(c.twice_coeff,inputs[c.left])!=dot(c.twice_coeff,inputs[c.right])||c.observed_left==c.observed_right)return 5;
      }
    }
    bool structured=sign(g0,0)*sign(g0,1)==sign(u0,0)*sign(u0,1)&&sign(g0,0)*sign(g0,1)==sign(g1,0)*sign(g1,1)&&sign(g0,0)*sign(g0,1)==sign(u1,0)*sign(u1,1);
    if(structured)predicted_structured++;
    if(structured!=(survivors>0))structured_mismatches++;
    if(survivors==0)no_carrier++;
    total_survivors+=survivors;survivors_hist[survivors]++;
    if(!nonadditive_emitted) {
      // Variables zero except two independent coordinates; rejects additive local table lifting.
      int z=42,xa=43,xb=46,xab=47;
      float lhs=value(truth[xab][0])+value(truth[z][0]);
      float rhs=value(truth[xa][0])+value(truth[xb][0]);
      if(bits(lhs)!=bits(rhs)) {
        printf("{\"kind\":\"nonadditive_partition_witness\",\"network\":");print_network(n);
        printf(",\"indices\":[42,43,46,47],\"inputs\":[");print_input(inputs[z]);printf(",");print_input(inputs[xa]);printf(",");print_input(inputs[xb]);printf(",");print_input(inputs[xab]);
        printf("],\"outputs\":[");print_output(truth[z]);printf(",");print_output(truth[xa]);printf(",");print_output(truth[xb]);printf(",");print_output(truth[xab]);
        printf("],\"rectangle_sum_bits\":[%u,%u]}\n",bits(lhs),bits(rhs));nonadditive_emitted=true;
      }
    }
    networks++;
  }
  printf("{\"kind\":\"census\",\"networks\":%llu,\"unique_numeric_truth_vectors\":%d,\"without_any_sufficient_E2M1_dot\":%llu,\"network_carrier_survivors\":%llu,\"injective_E2M1_carriers\":%llu}\n",(unsigned long long)networks,unique_truths,(unsigned long long)no_carrier,(unsigned long long)total_survivors,(unsigned long long)injective_carriers);
  for(auto [fibres,count]:fibres_hist)printf("{\"kind\":\"endpoint_fibre_count\",\"fibres\":%d,\"networks\":%llu}\n",fibres,(unsigned long long)count);
  for(auto [survivors,count]:survivors_hist)printf("{\"kind\":\"survivor_histogram\",\"carriers\":%d,\"networks\":%llu}\n",survivors,(unsigned long long)count);
  int uniform=0;
  for(int id=0;id<int(carriers.size());id++) {
    auto c=carriers[id];
    if(c.survivors==networks)uniform++;
    printf("{\"kind\":\"E2M1_carrier\",\"id\":%d,\"twice_coeff\":",id);print_input(c.twice_coeff);
    printf(",\"sufficient_networks\":%llu,\"rejection_network_id\":%d,\"left_index\":%d,\"right_index\":%d,\"left_input\":",(unsigned long long)c.survivors,c.network,c.left,c.right);
    print_input(inputs[c.left]);printf(",\"right_input\":");print_input(inputs[c.right]);
    printf(",\"left_observed\":");print_output(c.observed_left);printf(",\"right_observed\":");print_output(c.observed_right);printf("}\n");
    if(c.network==0) {
      if(dot(c.twice_coeff,inputs[c.left])!=dot(c.twice_coeff,inputs[c.right])||first_truth[c.left]==first_truth[c.right])return 2;
    }
  }
  for(int i=0;i<64;i++) {
    int label=dot({1,4,16},inputs[i])+42;
    if(label!=i)return 3;
    printf("{\"kind\":\"DP4A_radix4_truth\",\"index\":%d,\"input\":",i);print_input(inputs[i]);printf(",\"output\":");print_output(first_truth[i]);printf("}\n");
  }
  printf("{\"kind\":\"radix4_realizations\",\"coeff\":[1,4,16],\"bias\":42,\"DP4A_signed_weight_bytes\":[1,4,16,0],\"FP4_lane_positions\":[0,16,32],\"FP4_weights\":[1,1,1],\"weight_UE4M3_scales\":[1,4,16,1],\"FP4_common_activation_SF\":1,\"native_GPU_measured\":false,\"two_output_LUT_bytes\":512}\n");
  printf("{\"kind\":\"first_network\",\"network\":");print_network(first_network);printf("}\n");
  for(int label=0;label<32;label++)if(packed_reachable[label]) {
    printf("{\"kind\":\"E2M1_compound_truth\",\"twice_coeff\":[1,1,8],\"label\":%d,\"output\":",label);print_output(packed_truth[label]);printf("}\n");
  }
  printf("{\"kind\":\"structured_characterization_check\",\"predicted_networks\":%llu,\"counterexamples\":%llu}\n",(unsigned long long)predicted_structured,(unsigned long long)structured_mismatches);
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  printf("{\"kind\":\"terminal\",\"uniform_E2M1_carriers\":%d,\"DP4A_radix4_injective_points\":64,\"seconds\":%.6f}\n",uniform,seconds);
  if(std::ferror(stdout)) {std::fprintf(stderr,"error: result stream write failed\n");return 9;}
  return 0;
}
