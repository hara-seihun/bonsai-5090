#pragma once
#include "device.cuh"
namespace halo6000 {

// One CTA per contiguous block (grid-strided). Input coordinates are unsigned
// Sylvester order; output signs are applied AFTER normalized H, never before.
// All256 threads enter. scratch hasB floats; B128/256/512/4096 are supported.
template<int B,bool Quantize>
__device__ __forceinline__ void rotation_block_phase(const float*x,float*out,int rows,int K,const float*signs,int64_t sign_stride,float*scratch,ActivationBuffer a){
    constexpr int R=(B+255)/256;
    const int blocks=K/B;
    for(int unit=blockIdx.x;unit<rows*blocks;unit+=gridDim.x){
        int row=unit/blocks,block=unit%blocks,base=row*K+block*B;float v[R];
        #pragma unroll
        for(int r=0;r<R;r++){int i=threadIdx.x+256*r;v[r]=i<B?x[base+i]:0.f;}
        // Low five dimensions stay in warp registers.
        #pragma unroll
        for(int stride=1;stride<32;stride*=2){
            #pragma unroll
            for(int r=0;r<R;r++){float peer=__shfl_xor_sync(0xffffffff,v[r],stride);v[r]=(threadIdx.x&stride)?peer-v[r]:v[r]+peer;}
        }
        // Dimensions32/64/128 cross warps; exactly three shared exchanges max.
        #pragma unroll
        for(int stride=32;stride<B&&stride<256;stride*=2){
            #pragma unroll
            for(int r=0;r<R;r++){int i=threadIdx.x+256*r;if(i<B)scratch[i]=v[r];}
            __syncthreads();
            #pragma unroll
            for(int r=0;r<R;r++){int i=threadIdx.x+256*r;if(i<B){float peer=scratch[i^stride];v[r]=(i&stride)?peer-v[r]:v[r]+peer;}}
            __syncthreads();
        }
        // Larger dimensions mix registers owned by the same physical thread.
        #pragma unroll
        for(int stride=1;stride<R;stride*=2){
            #pragma unroll
            for(int p=0;p<R/2;p++){int lo=(p/stride)*(2*stride)+p%stride,hi=lo+stride;float l=v[lo],r=v[hi];v[lo]=l+r;v[hi]=l-r;}
        }
        const float norm=rsqrtf((float)B);
        #pragma unroll
        for(int r=0;r<R;r++){
            int i=threadIdx.x+256*r;
            if(i<B){float value=v[r]*norm;if(signs)value*=signs[row*sign_stride+block*B+i];out[base+i]=value;
                if constexpr(Quantize){float mx=warp_max(fabsf(value)),d=mx/127.f;a.qs[row*a.q_stride+block*B+i]=mx==0?0:int8_t(roundf(value/d));if((threadIdx.x&31)==0)a.scale[row*a.scale_stride+(block*B+i)/32]=__half2float(__float2half_rn(d));}
            }
        }
        __syncthreads(); // All consumers of block scratch finished before reuse.
    }
}

template<bool Quantize>
__device__ __forceinline__ void rotation_dispatch(const float*x,float*out,int rows,int K,int B,const float*signs,int64_t sign_stride,float*scratch,ActivationBuffer a){
    if(blockDim.x!=256){asm volatile("trap;");return;}
    if(B==0){ // Explicit flags0 control: bit-preserving copy, optional existing Q8 map.
        int lane=threadIdx.x&31,warp=blockIdx.x*8+threadIdx.x/32;
        for(int g=warp;g<rows*(K/32);g+=gridDim.x*8){int row=g/(K/32),local=g%(K/32),i=row*K+local*32+lane;float value=x[i];out[i]=value;
            if constexpr(Quantize){float mx=warp_max(fabsf(value)),d=mx/127.f;a.qs[row*a.q_stride+local*32+lane]=mx==0?0:int8_t(roundf(value/d));if(!lane)a.scale[row*a.scale_stride+local]=__half2float(__float2half_rn(d));}
        }
        return;
    }
    switch(B){
        case 128:rotation_block_phase<128,Quantize>(x,out,rows,K,signs,sign_stride,scratch,a);break;
        case 256:rotation_block_phase<256,Quantize>(x,out,rows,K,signs,sign_stride,scratch,a);break;
        case 512:rotation_block_phase<512,Quantize>(x,out,rows,K,signs,sign_stride,scratch,a);break;
        case 4096:rotation_block_phase<4096,Quantize>(x,out,rows,K,signs,sign_stride,scratch,a);break;
        default:asm volatile("trap;");
    }
}
__device__ __forceinline__ void rotation_phase(const float*x,float*out,int K,int B,const float*output_signs,float*shared_scratch){
    rotation_dispatch<false>(x,out,1,K,B,output_signs,0,shared_scratch,ActivationBuffer{});
}
__device__ __forceinline__ void rotation_q8_phase(const float*x,float*out,int K,int B,const float*output_signs,float*shared_scratch,ActivationBuffer q8){
    rotation_dispatch<true>(x,out,1,K,B,output_signs,0,shared_scratch,q8);
}
__device__ __forceinline__ void rotation_rows_phase(const float*x,float*out,int rows,int K,int B,const float*output_signs,int64_t sign_stride,float*shared_scratch){
    rotation_dispatch<false>(x,out,rows,K,B,output_signs,sign_stride,shared_scratch,ActivationBuffer{});
}
__device__ __forceinline__ void rotation_q8_rows_phase(const float*x,float*out,int rows,int K,int B,const float*output_signs,int64_t sign_stride,ActivationBuffer q8,float*shared_scratch){
    rotation_dispatch<true>(x,out,rows,K,B,output_signs,sign_stride,shared_scratch,q8);
}
} // namespace halo6000
