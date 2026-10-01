#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
// pv2-style: double-buffered smem tiles [2][16][128] for A and B, refilled from global each tile (LDG->reg, STS, bar)
__global__ void __launch_bounds__(256, 1) k(const uint4 * __restrict__ G, half2 * out, int ntl){
    __shared__ __align__(16) uint32_t As[2][16][128], Bs[2][16][128];
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    half2 h[8][8];
    for(int i=0;i<8;i++)for(int j=0;j<8;j++)h[i][j]=__float2half2_rn(0.f);
    const uint4 * g = G + (blockIdx.x%64)*1024*64;   // 64 different 1 MiB regions, mostly L2-missing
    uint4 ra[2], rb[2];
    auto gload=[&](int it){ for(int i=0;i<2;i++){ ra[i]=g[(it*1024+t+256*i)%65536]; rb[i]=g[(it*1024+512+t+256*i)%65536]; } };
    auto sstore=[&](int b){ for(int i=0;i<2;i++){ int l=t+256*i; *(uint4*)&As[b][l>>5][(l&31)*4]=ra[i]; *(uint4*)&Bs[b][l>>5][(l&31)*4]=rb[i]; } };
    auto tile=[&](int b){
#pragma unroll
        for(int k2=0;k2<16;k2++){
            uint4 a0=*(uint4*)&As[b][k2][ty*4], a1=*(uint4*)&As[b][k2][ty*4+64], b0=*(uint4*)&Bs[b][k2][tx*4], b1=*(uint4*)&Bs[b][k2][tx*4+64];
            const uint32_t a[8]={a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w}, bb[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
#pragma unroll
            for(int j=0;j<8;j++)
#pragma unroll
            for(int i=0;i<8;i++) h[i][j]=__hfma2(*(const half2*)&bb[j],*(const half2*)&a[i],h[i][j]);
        }
    };
    gload(0); sstore(0); __syncthreads();
    for(int it=0; it<ntl; it+=2){
        gload(it+1); tile(0); sstore(1); __syncthreads();
        gload(it+2); tile(1); sstore(0); __syncthreads();
    }
    half2 s=h[0][0]; for(int i=0;i<8;i++)for(int j=0;j<8;j++) s=__hadd2(s,h[i][j]);
    out[blockIdx.x*256+t]=s;
}
int main(){
    uint4*G; cudaMalloc(&G,64ull<<20); { size_t n=(64ull<<20)/2; uint16_t*hbuf=(uint16_t*)malloc(n*2); srand(1); for(size_t i=0;i<n;i++){ __half x=__float2half((rand()/(float)RAND_MAX-0.5f)*(getenv("AMP")?atof(getenv("AMP")):0.f)); hbuf[i]=*(uint16_t*)&x; } cudaMemcpy(G,hbuf,n*2,cudaMemcpyHostToDevice); } half2*o; cudaMalloc(&o,56*16*256*4);
    int nt=atoi(getenv("IT")); cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1); float ms;
    for(int r=0;r<3;r++){ cudaEventRecord(e0); k<<<56*8,256>>>(G,o,nt); cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms,e0,e1); printf("staged: %.2f TFLOPS (%s)\n",4.0*64*16*nt*256.0*56*8/ms/1e9,cudaGetErrorString(cudaGetLastError())); }
}
