// Fast loop for a prefill GEMM candidate: q6_K weights (real per-GPU slices) x f32 activations.
//   nvcc -O3 -arch=sm_60 -use_fast_math -lcublas -DCAND='"g1.cuh"' -o g harness.cu
//   ./g /mnt/fast/p100-scratch/f16mv/gate_8704x5120.q6k 8704 5120 1024
// Prints NMSE against an fp64 reference (on the first NREF columns) and GPU time for:
//   f16   today's path: dequant to half, x to half, cublasGemmEx COMPUTE_16F ALGO6, half out -> f32
//   f32   the accuracy mode: dequant to f32, cublasSgemm
//   cand  the candidate (cand_scratch/cand_run)
// Layout as ggml: W rows x K (K contiguous), X N x K, Y N x rows (rows contiguous).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_fp16.h>
#include <cublas_v2.h>
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };
static_assert(sizeof(block_q6_K)==210,"q6_K size");
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CB(x) do{cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS %s @%d: %d\n",#x,__LINE__,(int)e); exit(1);} }while(0)
#define NREF 64

// ggml's arithmetic: d (half->float) * sc * q, in float, then to T
template<typename T> __device__ __forceinline__ T to_t(float v);
template<> __device__ __forceinline__ float  to_t<float >(float v){return v;}
template<> __device__ __forceinline__ __half to_t<__half>(float v){return __float2half(v);}
template<> __device__ __forceinline__ double to_t<double>(float v){return (double)v;}
template<typename T> __global__ void k_deq(const block_q6_K* W, T* y){
  const block_q6_K& b=W[blockIdx.x]; int t=threadIdx.x; // 64 threads, 4 outputs each
  int ip=t/32, il=t%32, is=8*ip+il/16; const uint8_t*ql=b.ql+64*ip; uint8_t qh=b.qh[32*ip+il]; float d=__half2float(b.d);
  T* o=y+(size_t)blockIdx.x*256+128*ip+il;
  o[0]  =to_t<T>(d*b.scales[is+0]*(int)((int8_t)((ql[il]&0xF)|(((qh>>0)&3)<<4))-32));
  o[32] =to_t<T>(d*b.scales[is+2]*(int)((int8_t)((ql[il+32]&0xF)|(((qh>>2)&3)<<4))-32));
  o[64] =to_t<T>(d*b.scales[is+4]*(int)((int8_t)((ql[il]>>4)|(((qh>>4)&3)<<4))-32));
  o[96] =to_t<T>(d*b.scales[is+6]*(int)((int8_t)((ql[il+32]>>4)|(((qh>>6)&3)<<4))-32)); }
__global__ void k_f2h(const float*x,__half*y,size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) y[i]=__float2half(x[i]); }
__global__ void k_h2f(const __half*x,float*y,size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) y[i]=__half2float(x[i]); }
__global__ void k_ref(const double*W,const float*X,double*R,int rows,int K){ // R[n][r], n < NREF
  int r=blockIdx.x*blockDim.x+threadIdx.x, n=blockIdx.y; if(r>=rows) return; double s=0;
  for(int k=0;k<K;k++) s+=W[(size_t)r*K+k]*(double)X[(size_t)n*K+k]; R[(size_t)n*rows+r]=s; }

#include CAND   // size_t cand_scratch(int rows,int K,int N); void cand_run(const void*Wq,const float*X,float*Y,void*scr,int rows,int K,int N,cudaStream_t)

struct Ctx { const void*Wq; const float*X; float*Y; void*scr; __half*W16,*X16,*Y16; float*W32; int rows,K,N; cudaStream_t st; cublasHandle_t h; };
static void run_f16(Ctx&c){ int nb=c.rows*(c.K/256); k_deq<__half><<<nb,64,0,c.st>>>((const block_q6_K*)c.Wq,c.W16);
  size_t nx=(size_t)c.N*c.K; k_f2h<<<(nx+255)/256,256,0,c.st>>>(c.X,c.X16,nx);
  __half one=1.0f, zero=0.0f;
  CB(cublasGemmEx(c.h,CUBLAS_OP_T,CUBLAS_OP_N,c.rows,c.N,c.K,&one,c.W16,CUDA_R_16F,c.K,c.X16,CUDA_R_16F,c.K,&zero,c.Y16,CUDA_R_16F,c.rows,CUBLAS_COMPUTE_16F,CUBLAS_GEMM_ALGO6));
  size_t ny=(size_t)c.N*c.rows; k_h2f<<<(ny+255)/256,256,0,c.st>>>(c.Y16,c.Y,ny); }
static void run_f32(Ctx&c){ int nb=c.rows*(c.K/256); k_deq<float><<<nb,64,0,c.st>>>((const block_q6_K*)c.Wq,c.W32);
  float one=1,zero=0; CB(cublasSgemm(c.h,CUBLAS_OP_T,CUBLAS_OP_N,c.rows,c.N,c.K,&one,c.W32,c.K,c.X,c.K,&zero,c.Y,c.rows)); }
static void run_cand(Ctx&c){ cand_run(c.Wq,c.X,c.Y,c.scr,c.rows,c.K,c.N,c.st); }

int main(int argc,char**argv){
  if(argc<5){printf("usage: g W.q6k rows K N [outlier_frac=0.002] [reps=20] [which=all|cand]\n");return 1;}
  const char*fn=argv[1]; int rows=atoi(argv[2]),K=atoi(argv[3]),N=atoi(argv[4]); double of=argc>5?atof(argv[5]):0.002; int reps=argc>6?atoi(argv[6]):20;
  bool only_cand=argc>7&&argv[7][0]=='c';
  int nb=K/256; size_t wbytes=(size_t)rows*nb*210; std::vector<uint8_t> W(wbytes);
  FILE*f=fopen(fn,"rb"); if(!f||fread(W.data(),1,wbytes,f)!=wbytes){printf("read %s failed\n",fn);return 1;} fclose(f);
  std::mt19937 rng(42); std::normal_distribution<float> Nd(0,1); std::uniform_real_distribution<float> U(0,1);
  // per-channel (k) outliers, as in real activations: the same channels are large in every token
  std::vector<float> chs(K); for(auto&v:chs) v=U(rng)<of?60.f:1.f;
  std::vector<float> X((size_t)N*K); for(int n=0;n<N;n++) for(int k=0;k<K;k++) X[(size_t)n*K+k]=Nd(rng)*chs[k];
  Ctx c; c.rows=rows;c.K=K;c.N=N; CK(cudaStreamCreate(&c.st)); CB(cublasCreate(&c.h)); CB(cublasSetStream(c.h,c.st));
  void*dWq; float*dX; CK(cudaMalloc(&dWq,wbytes)); CK(cudaMalloc(&dX,X.size()*4));
  CK(cudaMemcpy(dWq,W.data(),wbytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));
  c.Wq=dWq; c.X=dX; size_t ny=(size_t)N*rows; CK(cudaMalloc(&c.Y,ny*4));
  CK(cudaMalloc(&c.W16,(size_t)rows*K*2)); CK(cudaMalloc(&c.X16,(size_t)N*K*2)); CK(cudaMalloc(&c.Y16,ny*2)); CK(cudaMalloc(&c.W32,(size_t)rows*K*4));
  size_t sc=cand_scratch(rows,K,N); CK(cudaMalloc(&c.scr,sc?sc:16));
  // fp64 reference on the first NREF columns
  double*dWd,*dR; CK(cudaMalloc(&dWd,(size_t)rows*K*8)); CK(cudaMalloc(&dR,(size_t)NREF*rows*8));
  k_deq<double><<<rows*nb,64,0,c.st>>>((const block_q6_K*)dWq,dWd); k_ref<<<dim3((rows+127)/128,NREF),128,0,c.st>>>(dWd,dX,dR,rows,K);
  std::vector<double> R((size_t)NREF*rows); CK(cudaMemcpy(R.data(),dR,R.size()*8,cudaMemcpyDeviceToHost)); cudaFree(dWd); cudaFree(dR);
  double r2=0; for(double v:R) r2+=v*v;
  struct P{const char*name; void(*fn)(Ctx&);} paths[3]={{"f16 ",run_f16},{"f32 ",run_f32},{"cand",run_cand}};
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for(auto&p:paths){ if(only_cand&&p.fn!=run_cand) continue;
    CK(cudaMemset(c.Y,0xff,ny*4)); p.fn(c); CK(cudaStreamSynchronize(c.st)); CK(cudaGetLastError());
    std::vector<float> Y(ny); CK(cudaMemcpy(Y.data(),c.Y,ny*4,cudaMemcpyDeviceToHost));
    double e=0,mx=0; int bad=0; for(size_t i=0;i<ny;i++) if(!std::isfinite(Y[i])) bad++;
    for(size_t i=0;i<R.size();i++){ double d=Y[i]-R[i]; e+=d*d; mx=std::max(mx,fabs(d)); }
    std::vector<float> t(reps); for(int i=0;i<3;i++) p.fn(c);
    for(int i=0;i<reps;i++){ cudaEventRecord(a,c.st); p.fn(c); cudaEventRecord(b,c.st); cudaEventSynchronize(b); cudaEventElapsedTime(&t[i],a,b); }
    std::sort(t.begin(),t.end()); double tf=2.0*rows*K*N/(t[reps/2]*1e-3)/1e12;
    printf("%s %dx%d N=%d | NMSE %.2e | nonfinite %d | ms median %.3f min %.3f | %.2f TFLOPS\n",p.name,rows,K,N,e/r2,bad,t[reps/2],t[0],tf); }
  return 0; }
