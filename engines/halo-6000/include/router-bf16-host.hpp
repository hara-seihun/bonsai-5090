#pragma once
#include "format.hpp"
#include <cstring>
#include <new>

namespace halo6000 {
// Host-only companion to router-bf16.cuh. No CUDA/libllama dependencies.
// This exact-storage API never rounds: an ineligible coefficient is Unsupported.
struct RouterBf16HostImage {
 static constexpr size_t LayerElements=size_t(Experts+1)*Hidden;
 static constexpr size_t LayerBytes=LayerElements*sizeof(uint16_t);
 static constexpr size_t ImageElements=size_t(Layers)*LayerElements;
 Provenance provenance;
 std::vector<uint16_t> coefficients; // [Layers][Experts+1][Hidden], router then shared
 size_t bytes()const{return coefficients.size()*sizeof(uint16_t);}
};
inline Result<RouterBf16HostImage> build_lossless_router_bf16(const HostImage&source){
 try {
  RouterBf16HostImage output;output.provenance=source.provenance;
  output.coefficients.resize(RouterBf16HostImage::ImageElements);
  for(int layer=0;layer<Layers;layer++)for(int shared=0;shared<2;shared++){
   const std::string name="blk."+std::to_string(layer)+(shared?".ffn_gate_inp_shexp.weight":".ffn_gate_inp.weight");
   const HostTensor*t=nullptr;
   for(const auto&candidate:source.tensors)if(candidate.name==name){
    if(t)return Error{ErrorCode::Geometry,"Duplicate router tensor: "+name};
    t=&candidate;
   }
   if(!t)return Error{ErrorCode::Geometry,"Missing router tensor: "+name};
   const std::vector<uint64_t>shape=shared?std::vector<uint64_t>{Hidden}:std::vector<uint64_t>{Hidden,Experts};
   if(t->format!=WeightFormat::F32)return Error{ErrorCode::Unsupported,"Lossless BF16 router builder requires loaded F32: "+name};
   const size_t elements=shared?Hidden:size_t(Hidden)*Experts;
   if(t->shape!=shape||t->bytes!=elements*4||!t->data)return Error{ErrorCode::Geometry,"Invalid loaded router geometry/extent: "+name};
   uint16_t*dest=output.coefficients.data()+size_t(layer)*RouterBf16HostImage::LayerElements+(shared?size_t(Experts)*Hidden:0);
   for(size_t i=0;i<elements;i++){
    uint32_t bits;std::memcpy(&bits,t->data+i*4,4);
    if((bits&0x7f800000u)==0x7f800000u||(bits&0xffffu)!=0)
     return Error{ErrorCode::Unsupported,"Router coefficient is not finite lossless BF16: "+name+" index "+std::to_string(i)};
    dest[i]=uint16_t(bits>>16); // Expansion <<16 recovers every original F32 bit.
   }
  }
  return output;
 }catch(const std::bad_alloc&){return Error{ErrorCode::Capacity,"Cannot allocate compact BF16 router image"};}
}
} // namespace halo6000
