// gemm-harness candidate: the shipped fold GEMM, optionally loaded from an edited cubin (env CUBIN).
//   nvcc -O3 -arch=sm_60 -lcublas -lcuda -DCAND='"../sass-gemm/cand_fold.cuh"' -o g ../gemm-harness/harness.cu
#include <cuda.h>
#include "fold_kernel.cuh"
static __device__ __forceinline__ float wrmax(float v){ for(int o=16;o;o>>=1) v=fmaxf(v,__shfl_xor_sync(~0u,v,o)); return v; }
__global__ void gemm_fold_prescale(const float * __restrict__ X, const int64_t s1, half * __restrict__ X16,
                                   float * __restrict__ cs, const int K, const int xexp) {
    const int n = blockIdx.x;
    const float * x = X + n*s1;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        m = fmaxf(m, fabsf(x[k]));
    }
    __shared__ float sm[32];
    m = wrmax(m);
    if ((threadIdx.x & 31) == 0) {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < blockDim.x/32 ? sm[threadIdx.x] : 0.0f;
        m = wrmax(m);
        if (threadIdx.x == 0) {
            sm[0] = m;
        }
    }
    __syncthreads();
    m = sm[0];
    const int   e = xexp < 0 ? 0 : m > 0.0f && isfinite(m) ? ilogbf(m) - xexp : 0;
    const float s = ldexpf(1.0f, -e);
    half * y = X16 + (int64_t) n*K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        y[k] = __float2half(x[k]*s);
    }
    if (threadIdx.x == 0) {
        cs[n] = ldexpf(1.0f, e) * 0x1p112f;
    }
}


static size_t cand_scratch(int rows,int K,int N){ return (size_t)rows*K*2 + (size_t)N*K*2 + (size_t)N*4 + 256; }
static void cand_run(const void*Wq,const float*X,float*Y,void*scr,int rows,int K,int N,cudaStream_t st){
  static CUfunction fn = nullptr; static bool init = false;
  if (!init) { init = true; const char * c = getenv("CUBIN");
    if (c) { CUmodule mod; if (cuModuleLoad(&mod, c) != CUDA_SUCCESS || cuModuleGetFunction(&fn, mod, "fold_gemm") != CUDA_SUCCESS) { printf("cubin load failed\n"); exit(1); } printf("using cubin %s\n", c); } }
  __half*W16=(__half*)scr; __half*X16=W16+(size_t)rows*K; float*cs=(float*)(X16+(size_t)N*K);
  k_deq<__half><<<rows*(K/256),64,0,st>>>((const block_q6_K*)Wq,W16);
  gemm_fold_prescale<<<N,256,0,st>>>(X,K,X16,cs,K,3);
  dim3 grid((rows+BM-1)/BM,(N+BN-1)/BN); int64_t sy=rows; int M=rows;
  if (fn) { const __half*w=W16; const __half*x=X16; const float*c=cs; void*args[]={&w,&x,&c,&Y,&M,&N,&K,&sy};
    if (cuLaunchKernel(fn,grid.x,grid.y,1,256,1,1,0,st,args,nullptr)!=CUDA_SUCCESS){printf("launch failed\n");exit(1);}
    static bool checked = false;
    if (!checked) {   // bitwise check against the compiled kernel, once
      checked = true; size_t ny=(size_t)N*rows; float*Y2; cudaMalloc(&Y2,ny*4);
      fold_gemm<<<grid,256,0,st>>>(W16,X16,cs,Y2,M,N,K,sy); cudaStreamSynchronize(st);
      std::vector<uint32_t> a(ny), b(ny); cudaMemcpy(a.data(),Y,ny*4,cudaMemcpyDeviceToHost); cudaMemcpy(b.data(),Y2,ny*4,cudaMemcpyDeviceToHost);
      size_t diff=0; for(size_t i=0;i<ny;i++) diff+=a[i]!=b[i];
      printf("bitwise vs compiled kernel: %zu of %zu outputs differ\n", diff, ny); cudaFree(Y2); } }
  else fold_gemm<<<grid,256,0,st>>>(W16,X16,cs,Y,M,N,K,sy); }
