#include "matmul.hpp"
#include <cassert>
#include <cstdio>
#include <cstddef>
using namespace halo6000;
static_assert(offsetof(PreparedMatmulView,operands)==72);
static_assert(offsetof(PreparedActivationView,operands)==48);
int main(){MatrixView w;w.n=16;w.k=128;w.format=WeightFormat::Pane2;assert(supported_matmul_source(w));auto b=matmul_weight_bytes(w);assert(b.operands==512&&b.scales==32);assert(matmul_activation_bytes(1,128)==1024);assert(matmul_activation_bytes(17,128,2)==6144);printf("{\"CPU_POD_passed\":true,\"PreparedMatmulView_bytes\":%zu,\"PreparedActivationView_bytes\":%zu}\n",sizeof(PreparedMatmulView),sizeof(PreparedActivationView));}
