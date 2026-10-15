#pragma once
#include "device.cuh"

namespace halo6000 {
struct AttentionBuffers {
    float *q_gate;   // [16][256 Q | 256 gate], projection output
    float *k;        // [2][256], overwritten by norm+RoPE
    float *v;        // [2][256], projection output
    float *q;        // [16][256], normalized+rotated
    float *scores;   // [16][state.context], logits then probabilities
    float *attended; // [16][256], attention result times sigmoid(gate)
};
// Matches the fork's host powf(base,-2/dim), rounded once toF32, then its
// device powf(theta_scale,pair_index). Not direct base**(-pair/32).
inline constexpr float AttentionRopeThetaScale=0x1.356656p-1f;
H6_HOST_DEVICE constexpr size_t attention_cache_offset(const StateView &s,int layer,int slot,int position,int head=0,int dim=0){
    return ((((size_t)layer*s.slots+slot)*s.context+position)*KVHeads+head)*HeadDim+dim;
}

// All phases require1D blocks with complete warps. The caller validates slot,
// attention_index0..9 and position0..context-1 and supplies grid.sync between
// phases. StateView.positions is untouched; caller owns advancement/reset.
__device__ __forceinline__ void attention_project_phase(const LayerWeights &w,ActivationView a,const float *normalized,AttentionBuffers b){
    matvec_phase(w.q,a,normalized,b.q_gate);
    matvec_phase(w.k,a,normalized,b.k);
    matvec_phase(w.v,a,normalized,b.v);
}
template<bool Append>
__device__ __forceinline__ void attention_norm_rope_impl(const LayerWeights &w,int position,AttentionBuffers b,StateView s,int layer,int slot){
    int lane=threadIdx.x&31,warp=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,stride=gridDim.x*(blockDim.x/32);
    for(int h=warp;h<QueryHeads+KVHeads;h+=stride){
        const bool isq=h<QueryHeads;int head=isq?h:h-QueryHeads;
        const float *src=isq?b.q_gate+head*(2*HeadDim):b.k+head*HeadDim;
        const float *gamma=isq?w.q_norm:w.k_norm;float *dst=isq?b.q+head*HeadDim:b.k+head*HeadDim;
        float val[HeadDim/32],sum=0;
        #pragma unroll
        for(int r=0;r<HeadDim/32;r++){val[r]=src[lane+32*r];sum=fmaf(val[r],val[r],sum);}
        float inv=rsqrtf(warp_sum(sum)/HeadDim+NormEpsilon);
        #pragma unroll
        for(int r=0;r<HeadDim/32;r++)val[r]=val[r]*inv*gamma[lane+32*r];
        const float angle=(float)position*powf(AttentionRopeThetaScale,(float)lane);
        float sn,cs;sincosf(angle,&sn,&cs);float x=val[0],y=val[1];
        val[0]=x*cs-y*sn;val[1]=x*sn+y*cs;
        #pragma unroll
        for(int r=0;r<HeadDim/32;r++)dst[lane+32*r]=val[r];
        if constexpr(Append){if(!isq){
            size_t base=attention_cache_offset(s,layer,slot,position,head);
            #pragma unroll
            for(int r=0;r<HeadDim/32;r++){
                int d=lane+32*r;*reinterpret_cast<half*>(s.key+base+d)=__float2half_rn(val[r]);
                *reinterpret_cast<half*>(s.value+base+d)=__float2half_rn(b.v[head*HeadDim+d]);
            }
        }}
    }
}
__device__ __forceinline__ void attention_norm_rope_phase(const LayerWeights &w,int position,AttentionBuffers b){
    attention_norm_rope_impl<false>(w,position,b,StateView{},0,0);
}
__device__ __forceinline__ void attention_norm_rope_append_phase(const LayerWeights &w,StateView s,int layer,int slot,int position,AttentionBuffers b){
    attention_norm_rope_impl<true>(w,position,b,s,layer,slot);
}
__device__ __forceinline__ void attention_append_phase(StateView s,int layer,int slot,int position,AttentionBuffers b){
    size_t base=attention_cache_offset(s,layer,slot,position);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<KVHeads*HeadDim;i+=gridDim.x*blockDim.x){
        *reinterpret_cast<half*>(s.key+base+i)=__float2half_rn(b.k[i]);
        *reinterpret_cast<half*>(s.value+base+i)=__float2half_rn(b.v[i]);
    }
}
__device__ __forceinline__ void attention_scores_phase(StateView s,int layer,int slot,int position,AttentionBuffers b){
    int lane=threadIdx.x&31,warp=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,stride=gridDim.x*(blockDim.x/32),length=position+1;
    for(int unit=warp;unit<QueryHeads*length;unit+=stride){
        int head=unit/length,t=unit%length,kh=head/(QueryHeads/KVHeads);float sum=0;
        const float *q=b.q+head*HeadDim;const uint16_t *key=s.key+attention_cache_offset(s,layer,slot,t,kh);
        #pragma unroll
        for(int r=0;r<HeadDim/32;r++){int d=lane+32*r;sum=fmaf(q[d],half_value(key+d),sum);}
        sum=warp_sum(sum)*AttentionScale;if(!lane)b.scores[(size_t)head*s.context+t]=sum;
    }
}
__device__ __forceinline__ void attention_softmax_phase(StateView s,int position,AttentionBuffers b){
    int lane=threadIdx.x&31,warp=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,stride=gridDim.x*(blockDim.x/32),length=position+1;
    for(int head=warp;head<QueryHeads;head+=stride){
        float *score=b.scores+(size_t)head*s.context,maxval=-INFINITY;
        for(int t=lane;t<length;t+=32)maxval=fmaxf(maxval,score[t]);maxval=warp_max(maxval);
        float sum=0;for(int t=lane;t<length;t+=32){float e=expf(score[t]-maxval);score[t]=e;sum+=e;}
        float inv=1.f/warp_sum(sum);for(int t=lane;t<length;t+=32)score[t]*=inv;
    }
}
__device__ __forceinline__ void attention_value_gate_phase(StateView s,int layer,int slot,int position,AttentionBuffers b){
    // Four adjacent dimensions x eight time partitions per warp keeps value
    // accesses partly coalesced without an extra partial-sum workspace.
    int lane=threadIdx.x&31,warp=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,stride=gridDim.x*(blockDim.x/32);
    for(int unit=warp;unit<QueryHeads*HeadDim/4;unit+=stride){
        int i=unit*4+(lane&3),head=i/HeadDim,dim=i%HeadDim,kh=head/(QueryHeads/KVHeads);float sum=0;
        const float *prob=b.scores+(size_t)head*s.context;
        const uint16_t *value=s.value+attention_cache_offset(s,layer,slot,0,kh,dim);
        for(int t=lane/4;t<=position;t+=8)sum=fmaf(prob[t],half_value(value+(size_t)t*KVHeads*HeadDim),sum);
        #pragma unroll
        for(int delta=4;delta<=16;delta*=2)sum+=__shfl_xor_sync(0xffffffff,sum,delta);
        if(lane<4)b.attended[i]=sum*sigmoid(b.q_gate[head*(2*HeadDim)+HeadDim+dim]);
    }
}
// Producer coarsening: eight time-partition warps cooperate on32 adjacent
// values. The shared reduction preserves split value_gate's exact butterfly
// order, then warp0 applies the unchanged quantize_phase map. All256 threads
// enter; the1KiB scratch is block-local, not additional sequence state.
__device__ __forceinline__ void attention_value_gate_q8_phase(StateView s,int layer,int slot,int position,AttentionBuffers b,ActivationBuffer a){
    __shared__ float partial[256];
    if(blockDim.x!=256){asm volatile("trap;");return;}
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    for(int group=blockIdx.x;group<QueryHeads*HeadDim/32;group+=gridDim.x){
        int i=group*32+lane,head=i/HeadDim,dim=i%HeadDim,kh=head/(QueryHeads/KVHeads);float sum=0;
        const float *prob=b.scores+(size_t)head*s.context;
        const uint16_t *value=s.value+attention_cache_offset(s,layer,slot,0,kh,dim);
        for(int t=warp;t<=position;t+=8)sum=fmaf(prob[t],half_value(value+(size_t)t*KVHeads*HeadDim),sum);
        partial[threadIdx.x]=sum;__syncthreads();
        if(!warp){
            float s01=partial[lane]+partial[lane+32],s23=partial[lane+64]+partial[lane+96];
            float s45=partial[lane+128]+partial[lane+160],s67=partial[lane+192]+partial[lane+224];
            float x=((s01+s23)+(s45+s67))*sigmoid(b.q_gate[head*(2*HeadDim)+HeadDim+dim]);
            b.attended[i]=x;float mx=warp_max(fabsf(x)),d=mx/127.f;
            a.qs[i]=mx==0?0:int8_t(roundf(x/d));if(!lane)a.scale[group]=__half2float(__float2half_rn(d));
        }
        __syncthreads();
    }
}
} // namespace halo6000
