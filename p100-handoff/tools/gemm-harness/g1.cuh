// Candidate 1: fp16 HFMA2 GEMM with fp32 folds.
// W dequantized to half (as today), X converted to half after a per-column power-of-2 prescale.
// Each thread holds an 8x8 output tile as half2 accumulators (lanes = even/odd k); every FOLD k2
// steps the two lanes are added (HADD2, one rounding) and moved into fp32 accumulators with an
// integer half->float conversion (arith shift + mask; the 2^112 rebias rides in the FFMA).
#ifndef FOLD
#define FOLD 16
#endif
#define BM 128
#define BN 128
#define BK2 16   // k2 (half2) steps per tile = 32 k
__global__ void g1_prescale(const float* X, __half* X16, float* cs, int K){
  int n=blockIdx.x; const float* x=X+(size_t)n*K; float m=0;
  for(int k=threadIdx.x;k<K;k+=blockDim.x) m=fmaxf(m,fabsf(x[k]));
  __shared__ float sm[32]; for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(~0u,m,o));
  if((threadIdx.x&31)==0) sm[threadIdx.x>>5]=m; __syncthreads();
  if(threadIdx.x<32){ m=threadIdx.x<blockDim.x/32?sm[threadIdx.x]:0; for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(~0u,m,o)); if(!threadIdx.x) sm[0]=m; }
  __syncthreads(); m=sm[0]; int e=m>0?ilogbf(m)-3:0; float s=ldexpf(1.f,-e);
  __half* y=X16+(size_t)n*K; for(int k=threadIdx.x;k<K;k+=blockDim.x) y[k]=__float2half(x[k]*s);
  if(!threadIdx.x) cs[n]=ldexpf(1.f,e)*0x1p112f; }   // 2^112 undoes the conversion's rebias (applied at the end)

static __device__ __forceinline__ float fold_h(__half2 p){
  __half2 s=__hadd2(p,__lowhigh2highlow(p));             // hi lane = lo+hi
  uint32_t u=*(uint32_t*)&s; int32_t v=((int32_t)u)>>3;  // hi half: sign smeared into 28..31, e/m at 13..27
  return __int_as_float(v & 0x8FFFE000); }                // = hi * 2^-112 (normals), FTZ for subnormals

__global__ void __launch_bounds__(256,1) g1_gemm(const __half* __restrict__ W,const __half* __restrict__ X,const float* __restrict__ cs,float* __restrict__ Y,int M,int N,int K){
  __shared__ __align__(16) uint32_t As[2][BK2][BM], Bs[2][BK2][BN];
  const int t=threadIdx.x, tx=t&15, ty=t>>4, m0=blockIdx.x*BM, n0=blockIdx.y*BN;
  float acc[8][8]; __half2 h[8][8];
  #pragma unroll
  for(int i=0;i<8;i++) for(int j=0;j<8;j++){acc[i][j]=0; h[i][j]=__float2half2_rn(0.f);}
  uint4 ra[2],rb[2];
  auto gload=[&](int k0){
    #pragma unroll
    for(int i=0;i<2;i++){ int l=t+256*i, r=l>>2, c=l&3;
      ra[i]= m0+r<M ? *(const uint4*)(W+(size_t)(m0+r)*K+k0+c*8) : make_uint4(0,0,0,0);
      rb[i]= n0+r<N ? *(const uint4*)(X+(size_t)(n0+r)*K+k0+c*8) : make_uint4(0,0,0,0); } };
  auto sstore=[&](int buf){
    #pragma unroll
    for(int i=0;i<2;i++){ int l=t+256*i, r=l>>2, c=l&3;
      As[buf][c*4+0][r]=ra[i].x; As[buf][c*4+1][r]=ra[i].y; As[buf][c*4+2][r]=ra[i].z; As[buf][c*4+3][r]=ra[i].w;
      Bs[buf][c*4+0][r]=rb[i].x; Bs[buf][c*4+1][r]=rb[i].y; Bs[buf][c*4+2][r]=rb[i].z; Bs[buf][c*4+3][r]=rb[i].w; } };
  const int nt=K/(2*BK2);
  gload(0); sstore(0); __syncthreads();
  for(int it=0;it<nt;it++){
    const int buf=it&1;
    if(it+1<nt) gload((it+1)*2*BK2);
    const bool restart=(it*BK2)%FOLD==0;
    #pragma unroll
    for(int k2=0;k2<BK2;k2++){
      uint4 a0=*(const uint4*)&As[buf][k2][ty*4], a1=*(const uint4*)&As[buf][k2][64+ty*4];
      uint4 b0=*(const uint4*)&Bs[buf][k2][tx*4], b1=*(const uint4*)&Bs[buf][k2][64+tx*4];
      uint32_t a[8]={a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w}, b[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
      #pragma unroll
      for(int i=0;i<8;i++)
        #pragma unroll
        for(int j=0;j<8;j++) h[i][j]= (k2==0 && restart) ? __hmul2(*(__half2*)&a[i],*(__half2*)&b[j]) : __hfma2(*(__half2*)&a[i],*(__half2*)&b[j],h[i][j]);
    }
    if(((it+1)*BK2)%FOLD==0 || it+1==nt){
      #pragma unroll
      for(int i=0;i<8;i++)
        #pragma unroll
        for(int j=0;j<8;j++){ acc[i][j]+=fold_h(h[i][j]); }
    }
    if(it+1<nt){ sstore(buf^1); }
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<8;j++){ int n=n0+(j<4?tx*4+j:64+tx*4+j-4); if(n>=N) continue; float s=cs[n];
    #pragma unroll
    for(int ih=0;ih<2;ih++){ int m=m0+ih*64+ty*4; if(m>=M) continue;
      *(float4*)(Y+(size_t)n*M+m)=make_float4(acc[ih*4+0][j]*s,acc[ih*4+1][j]*s,acc[ih*4+2][j]*s,acc[ih*4+3][j]*s); } } }

size_t cand_scratch(int rows,int K,int N){ return (size_t)rows*K*2+(size_t)N*K*2+(size_t)N*4; }
void cand_run(const void*Wq,const float*X,float*Y,void*scr,int rows,int K,int N,cudaStream_t st){
  __half* W16=(__half*)scr; __half* X16=W16+(size_t)rows*K; float* cs=(float*)(X16+(size_t)N*K);
  k_deq<__half><<<rows*(K/256),64,0,st>>>((const block_q6_K*)Wq,W16);
  g1_prescale<<<N,256,0,st>>>(X,X16,cs,K);
  g1_gemm<<<dim3((rows+BM-1)/BM,(N+BN-1)/BN),256,0,st>>>(W16,X16,cs,Y,rows,N,K); }
