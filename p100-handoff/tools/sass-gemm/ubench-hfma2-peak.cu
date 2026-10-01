#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
template<int NACC>
__global__ void k(half2 * out, half2 a0, half2 b0, int iters){
    half2 acc[NACC]; half2 a[4], b[4];
    for (int i=0;i<4;i++){ a[i]=__hadd2(a0,__float2half2_rn(threadIdx.x*1e-3f+i)); b[i]=__hadd2(b0,__float2half2_rn(i*1e-3f));}
    for (int i=0;i<NACC;i++) acc[i]=__float2half2_rn(0.f);
    for (int it=0; it<iters; it++){
#pragma unroll
        for (int j=0;j<4;j++)
#pragma unroll
        for (int i=0;i<NACC/4;i++) acc[j*(NACC/4)+i]=__hfma2(b[j],a[i%4],acc[j*(NACC/4)+i]);
    }
    half2 s=acc[0]; for(int i=1;i<NACC;i++) s=__hadd2(s,acc[i]);
    out[blockIdx.x*blockDim.x+threadIdx.x]=s;
}
int main(){
    half2 *o; cudaMalloc(&o, 56*64*1024*4);
    int iters=atoi(getenv("IT")); cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int thr : {128, 256, 512}) for (int bl : {56*4, 56*8}) {
        k<32><<<bl,thr>>>(o,__float2half2_rn(1e-4f),__float2half2_rn(1e-4f),100);
        cudaEventRecord(e0); k<32><<<bl,thr>>>(o,__float2half2_rn(1e-4f),__float2half2_rn(1e-4f),iters); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms,e0,e1);
        double fl=4.0*32*iters*(double)thr*bl; printf("thr %d blocks %d: %.2f TFLOPS\n",thr,bl,fl/ms/1e9);
    }
}
