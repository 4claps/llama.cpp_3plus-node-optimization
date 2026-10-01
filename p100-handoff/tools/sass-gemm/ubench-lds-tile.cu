#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
// 8x8 half2 tile per thread from smem fragments (2 LDS.128 a + 2 LDS.128 b per k2), like fa_fold
template<int PF>
__global__ void __launch_bounds__(256, 2) k(half2 * out, int iters){
    __shared__ __align__(16) uint32_t As[16][128], Bs[16][128];
    for (int i=threadIdx.x;i<16*128;i+=256){ As[0][i]=0x00010001u*(i&7); Bs[0][i]=0x00010001u*(i&3);} __syncthreads();
    const int tx=threadIdx.x&15, ty=threadIdx.x>>4;
    half2 h[8][8];
    for(int i=0;i<8;i++)for(int j=0;j<8;j++)h[i][j]=__float2half2_rn(0.f);
    for (int it=0; it<iters; it++){
        uint4 a0n=*(uint4*)&As[0][ty*4], a1n=*(uint4*)&As[0][ty*4+64], b0n=*(uint4*)&Bs[0][tx*4], b1n=*(uint4*)&Bs[0][tx*4+64];
#pragma unroll
        for(int k2=0;k2<16;k2++){
            uint4 a0=a0n,a1=a1n,b0=b0n,b1=b1n;
            if (PF && k2<15){ a0n=*(uint4*)&As[k2+1][ty*4]; a1n=*(uint4*)&As[k2+1][ty*4+64]; b0n=*(uint4*)&Bs[k2+1][tx*4]; b1n=*(uint4*)&Bs[k2+1][tx*4+64]; }
            if (!PF && k2>0){ a0=*(uint4*)&As[k2][ty*4]; a1=*(uint4*)&As[k2][ty*4+64]; b0=*(uint4*)&Bs[k2][tx*4]; b1=*(uint4*)&Bs[k2][tx*4+64]; }
            const uint32_t a[8]={a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w}, b[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
#pragma unroll
            for(int j=0;j<8;j++)
#pragma unroll
            for(int i=0;i<8;i++) h[i][j]=__hfma2(*(const half2*)&b[j],*(const half2*)&a[i],h[i][j]);
        }
    }
    half2 s=h[0][0]; for(int i=0;i<8;i++)for(int j=0;j<8;j++) s=__hadd2(s,h[i][j]);
    out[blockIdx.x*256+threadIdx.x]=s;
}
int main(){
    half2*o; cudaMalloc(&o,56*64*256*4); int it=atoi(getenv("IT")); cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1); float ms;
    for(int pf=0;pf<2;pf++) for(int r=0;r<2;r++){
        cudaEventRecord(e0); if(pf) k<1><<<56*16,256,getenv("SM")?atoi(getenv("SM")):0>>>(o,it); else k<0><<<56*16,256,getenv("SM")?atoi(getenv("SM")):0>>>(o,it); cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms,e0,e1); printf("pf %d: %.2f TFLOPS (%s)\n",pf,4.0*64*16*it*256.0*56*16/ms/1e9, cudaGetErrorString(cudaGetLastError()));
    }
}
