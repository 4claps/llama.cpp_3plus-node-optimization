// Candidate 2: g1 with a configurable block: BM=128 x BN (64 or 128), NT = BN/8*16 threads,
// so several blocks share an SM and hide each other's barriers. Fold every FOLD k2 steps.
#ifndef FOLD
#define FOLD 32
#endif
#ifndef BN
#define BN 64
#endif
#ifndef MINB
#define MINB 2
#endif
#define BM 128
#define BK2 16
#define NTX (BN/8)
#define NT (NTX*16)
__global__ void g1_prescale(const float* X, __half* X16, float* cs, int K){
  int n=blockIdx.x; const float* x=X+(size_t)n*K; float m=0;
  for(int k=threadIdx.x;k<K;k+=blockDim.x) m=fmaxf(m,fabsf(x[k]));
  __shared__ float sm[32]; for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(~0u,m,o));
  if((threadIdx.x&31)==0) sm[threadIdx.x>>5]=m; __syncthreads();
  if(threadIdx.x<32){ m=threadIdx.x<blockDim.x/32?sm[threadIdx.x]:0; for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(~0u,m,o)); if(!threadIdx.x) sm[0]=m; }
  __syncthreads(); m=sm[0]; int e=m>0?ilogbf(m)-3:0; float s=ldexpf(1.f,-e);
  __half* y=X16+(size_t)n*K; for(int k=threadIdx.x;k<K;k+=blockDim.x) y[k]=__float2half(x[k]*s);
  if(!threadIdx.x) cs[n]=ldexpf(1.f,e)*0x1p112f; }
static __device__ __forceinline__ float fold_h(__half2 p){
  __half2 s=__hadd2(p,__lowhigh2highlow(p)); uint32_t u=*(uint32_t*)&s; int32_t v=((int32_t)u)>>3;
  return __int_as_float(v & 0x8FFFE000); }
__global__ void __launch_bounds__(NT,MINB) g2_gemm(const __half* __restrict__ W,const __half* __restrict__ X,const float* __restrict__ cs,float* __restrict__ Y,int M,int N,int K){
  __shared__ __align__(16) uint32_t As[2][BK2][BM], Bs[2][BK2][BN];
  const int t=threadIdx.x, tx=t%NTX, ty=t/NTX, m0=blockIdx.x*BM, n0=blockIdx.y*BN;
  float acc[8][8]; __half2 h[8][8];
  #pragma unroll
  for(int i=0;i<8;i++) for(int j=0;j<8;j++){acc[i][j]=0; h[i][j]=__float2half2_rn(0.f);}
  constexpr int LA=BM*4/NT, LB=BN*4/NT;   // 16B chunks per thread
  uint4 ra[LA],rb[LB];
  auto gload=[&](int k0){
    #pragma unroll
    for(int i=0;i<LA;i++){ int l=t+NT*i, r=l>>2, c=l&3; ra[i]= m0+r<M ? *(const uint4*)(W+(size_t)(m0+r)*K+k0+c*8) : make_uint4(0,0,0,0); }
    #pragma unroll
    for(int i=0;i<LB;i++){ int l=t+NT*i, r=l>>2, c=l&3; rb[i]= n0+r<N ? *(const uint4*)(X+(size_t)(n0+r)*K+k0+c*8) : make_uint4(0,0,0,0); } };
  auto sstore=[&](int buf){
    #pragma unroll
    for(int i=0;i<LA;i++){ int l=t+NT*i, r=l>>2, c=l&3; const int rs=r^(c<<3); As[buf][c*4+0][rs]=ra[i].x; As[buf][c*4+1][rs]=ra[i].y; As[buf][c*4+2][rs]=ra[i].z; As[buf][c*4+3][rs]=ra[i].w; }
    #pragma unroll
    for(int i=0;i<LB;i++){ int l=t+NT*i, r=l>>2, c=l&3; const int rs=r^(c<<3); Bs[buf][c*4+0][rs]=rb[i].x; Bs[buf][c*4+1][rs]=rb[i].y; Bs[buf][c*4+2][rs]=rb[i].z; Bs[buf][c*4+3][rs]=rb[i].w; } };
  const int nt=K/(2*BK2);
  gload(0); sstore(0); __syncthreads();
  for(int it=0;it<nt;it++){
    const int buf=it&1;
#ifndef NOLOAD
    if(it+1<nt) gload((it+1)*2*BK2);
#endif
    const bool restart=(it*BK2)%FOLD==0;
    #pragma unroll
    for(int k2=0;k2<BK2;k2++){
      const int sw=(k2>>2)<<3; uint4 a0=*(const uint4*)&As[buf][k2][(ty*4)^sw], a1=*(const uint4*)&As[buf][k2][(64+ty*4)^sw];
      uint4 b0=*(const uint4*)&Bs[buf][k2][(tx*4)^sw], b1=*(const uint4*)&Bs[buf][k2][(BN/2+tx*4)^sw];
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
        for(int j=0;j<8;j++) acc[i][j]+=fold_h(h[i][j]);
    }
#ifndef NOLOAD
    if(it+1<nt){ sstore(buf^1); }
#endif
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<8;j++){ int n=n0+(j<4?tx*4+j:BN/2+tx*4+j-4); if(n>=N) continue; float s=cs[n];
    #pragma unroll
    for(int ih=0;ih<2;ih++){ int m=m0+ih*64+ty*4; if(m>=M) continue;
      *(float4*)(Y+(size_t)n*M+m)=make_float4(acc[ih*4+0][j]*s,acc[ih*4+1][j]*s,acc[ih*4+2][j]*s,acc[ih*4+3][j]*s); } } }
size_t cand_scratch(int rows,int K,int N){ return (size_t)rows*K*2+(size_t)N*K*2+(size_t)N*4; }
void cand_run(const void*Wq,const float*X,float*Y,void*scr,int rows,int K,int N,cudaStream_t st){
  __half* W16=(__half*)scr; __half* X16=W16+(size_t)rows*K; float* cs=(float*)(X16+(size_t)N*K);
  k_deq<__half><<<rows*(K/256),64,0,st>>>((const block_q6_K*)Wq,W16);
  g1_prescale<<<N,256,0,st>>>(X,X16,cs,K);
  if(!getenv("ONLYPRE")) g2_gemm<<<dim3((rows+BM-1)/BM,(N+BN-1)/BN),NT,0,st>>>(W16,X16,cs,Y,rows,N,K); }
