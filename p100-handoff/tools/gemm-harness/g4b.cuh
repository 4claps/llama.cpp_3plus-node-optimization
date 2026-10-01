// Candidate 4: smem rows K-contiguous (row stride 20 words: conflict-free LDS.128 / STS.128),
// thread tile rows m = ty + 16i, cols n = tx + 16j; fragments read 4 k2 at a time.
#ifndef FOLD
#define FOLD 64
#endif
#define BM 128
#define BN 128
#define BK2 16
#define RS 20        // smem row stride in words (16 k2 + 4 pad)
#define NT 256
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
__global__ void __launch_bounds__(NT,1) g4_gemm(const __half* __restrict__ W,const __half* __restrict__ X,const float* __restrict__ cs,float* __restrict__ Y,int M,int N,int K){
  __shared__ __align__(16) uint32_t As[2][BM*RS], Bs[2][BN*RS];
  const int t=threadIdx.x, ty=t&15, tx=t>>4, m0=blockIdx.x*BM, n0=blockIdx.y*BN;
  float acc[8][8]; __half2 h[8][8];
  #pragma unroll
  for(int i=0;i<8;i++) for(int j=0;j<8;j++){acc[i][j]=0; h[i][j]=__float2half2_rn(0.f);}
  uint4 ra[2],rb[2];
  auto gload=[&](int k0){
    #pragma unroll
    for(int i=0;i<2;i++){ int l=t+NT*i, r=l>>2, c=l&3;
      ra[i]= m0+r<M ? *(const uint4*)(W+(size_t)(m0+r)*K+k0+c*8) : make_uint4(0,0,0,0);
      rb[i]= n0+r<N ? *(const uint4*)(X+(size_t)(n0+r)*K+k0+c*8) : make_uint4(0,0,0,0); } };
  auto sstore=[&](int buf){
    #pragma unroll
    for(int i=0;i<2;i++){ int l=t+NT*i, r=l>>2, c=l&3;
      *(uint4*)&As[buf][r*RS+c*4]=ra[i]; *(uint4*)&Bs[buf][r*RS+c*4]=rb[i]; } };
  const int nt=K/(2*BK2);
  gload(0); sstore(0); __syncthreads();
  for(int it=0;it<nt;it++){
    const int buf=it&1;
    if(it+1<nt) gload((it+1)*2*BK2);
    const bool restart=(it*BK2)%FOLD==0;
    #pragma unroll
    for(int kq=0;kq<BK2/4;kq++){
      uint4 a[8],b[8];
      #pragma unroll
      for(int i=0;i<8;i++){ a[i]=*(const uint4*)&As[buf][(ty+16*i)*RS+kq*4]; b[i]=*(const uint4*)&Bs[buf][(tx+16*i)*RS+kq*4]; }
      #pragma unroll
      for(int kk=0;kk<4;kk++){
        #pragma unroll
        for(int i=0;i<8;i++){
          const uint32_t ai=kk==0?a[i].x:kk==1?a[i].y:kk==2?a[i].z:a[i].w;
          #pragma unroll
          for(int j=0;j<8;j++){
            const uint32_t bj=kk==0?b[j].x:kk==1?b[j].y:kk==2?b[j].z:b[j].w;
            h[i][j]= (kq==0 && kk==0 && restart) ? __hmul2(*(const __half2*)&ai,*(const __half2*)&bj) : __hfma2(*(const __half2*)&ai,*(const __half2*)&bj,h[i][j]); } } } }
    if(((it+1)*BK2)%FOLD==0 || it+1==nt){
      #pragma unroll
      for(int i=0;i<8;i++)
        #pragma unroll
        for(int j=0;j<8;j++) acc[i][j]+=fold_h(h[i][j]);
    }
    if(it+1<nt){ sstore(buf^1); }
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<8;j++){ int n=n0+tx+16*j; if(n>=N) continue; float s=cs[n];
    #pragma unroll
    for(int i=0;i<8;i++){ int m=m0+ty+16*i; if(m<M) Y[(size_t)n*M+m]=acc[i][j]*s; } } }
size_t cand_scratch(int rows,int K,int N){ return (size_t)rows*K*2+(size_t)N*K*2+(size_t)N*4; }
void cand_run(const void*Wq,const float*X,float*Y,void*scr,int rows,int K,int N,cudaStream_t st){
  __half* W16=(__half*)scr; __half* X16=W16+(size_t)rows*K; float* cs=(float*)(X16+(size_t)N*K);
  k_deq<__half><<<rows*(K/256),64,0,st>>>((const block_q6_K*)Wq,W16);
  g1_prescale<<<N,256,0,st>>>(X,X16,cs,K);
  g4_gemm<<<dim3((rows+BM-1)/BM,(N+BN-1)/BN),NT,0,st>>>(W16,X16,cs,Y,rows,N,K); }
