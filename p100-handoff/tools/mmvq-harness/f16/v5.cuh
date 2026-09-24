// v5: fp16 q6_K multi-column matvec, rebuilt for instruction count.
// Requires an even number of q6_K blocks per row (all per-GPU shapes here), so every row and every
// NBF-block window starts 4-byte aligned; block b's 2-byte phase is then b&1, known at compile
// time, and every field read is one aligned LDS (even b) or two plus a PRMT (odd b).
// Numerics: x -> half with a power-of-2 prescale per (column, window) so |x| < 1; weight
// w = (d*1024) * sc * (q-32) in half (one rounding of the weight); per lane a fused fp16 chain over
// the window (4 HFMA2 per block), folded into fp32 once per window.
#ifndef V5_NW
#define V5_NW 4
#endif
#ifndef V5_RPW
#define V5_RPW 4
#endif
#ifndef V5_NBF
#define V5_NBF 4
#endif
constexpr int V5_WIN=V5_NBF*256;
__global__ void v5_prep(const float* __restrict__ X, __half* __restrict__ XS, float* __restrict__ S, int K){
  const int w=blockIdx.x*blockDim.y+threadIdx.y, c=blockIdx.y, nw=(K+V5_WIN-1)/V5_WIN; if(w>=nw) return;
  const float* x=X+(size_t)c*K+(size_t)w*V5_WIN; constexpr int PER=V5_WIN/32;
  const int len=min(V5_WIN,K-w*V5_WIN);
  float v[PER]; float m=0;
  #pragma unroll
  for(int j=0;j<PER/4;j++){ float4 f=make_float4(0,0,0,0); if(j*128<len) f=__ldg((const float4*)x+j*32+threadIdx.x); v[4*j]=f.x; v[4*j+1]=f.y; v[4*j+2]=f.z; v[4*j+3]=f.w;
    m=fmaxf(m,fmaxf(fmaxf(fabsf(f.x),fabsf(f.y)),fmaxf(fabsf(f.z),fabsf(f.w)))); }
  for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(~0u,m,o));
  int e=0; if(m>0) frexpf(m,&e); const float inv=ldexpf(1.f,-e);
  __half* xs=XS+(size_t)c*K+(size_t)w*V5_WIN;
  #pragma unroll
  for(int j=0;j<PER/4;j++){ if(j*128>=len) break; __half2 a=__floats2half2_rn(v[4*j]*inv,v[4*j+1]*inv), b=__floats2half2_rn(v[4*j+2]*inv,v[4*j+3]*inv);
    uint2 u; u.x=*(uint32_t*)&a; u.y=*(uint32_t*)&b; ((uint2*)xs)[j*32+threadIdx.x]=u; }
  if(threadIdx.x==0) S[c*nw+w]=ldexpf(1.f,e-10);
}
// 32-bit field at byte offset OFF of block B (relative to a 4-aligned window base)
template<int B,int OFF> static __device__ __forceinline__ uint32_t fld(const uint32_t* wb, int lane_words){
  constexpr int A=210*B+OFF;
  if constexpr ((A&3)==0) return wb[A/4+lane_words];
  else return __byte_perm(wb[A/4+lane_words],wb[A/4+1+lane_words],0x5432);
}
template<int NC, int NBLK, int B=0>
static __device__ __forceinline__ void v5_blocks(const uint32_t* const* wb, const __half* const* xw, __half2 (&t)[NC][V5_RPW],
    const int P, const int ql_w, const int qh_w, const int vh_shift, const int sc_we, const int sc_wo, const uint32_t sc_sel_e, const uint32_t sc_sel_o){
  if constexpr (B<NBLK){
    const __half2 k1056=__float2half2_rn(1056.f), k1152=__float2half2_rn(1152.f), k1024=__float2half2_rn(1024.f);
    __half2 xa[NC][2], xb[NC][2];
    #pragma unroll
    for(int c=0;c<NC;c++){ const __half* xs=xw[c]+B*256;
      const uint2 a=*(const uint2*)(xs+P), bb=*(const uint2*)(xs+P+64);
      xa[c][0]=*(const __half2*)&a.x; xa[c][1]=*(const __half2*)&a.y; xb[c][0]=*(const __half2*)&bb.x; xb[c][1]=*(const __half2*)&bb.y; }
    #pragma unroll
    for(int i=0;i<V5_RPW;i++){
      const uint32_t vl=fld<B,0>(wb[i],ql_w);
      const uint32_t vh=fld<B,128>(wb[i],qh_w)>>vh_shift;
      // scales so and so+4: window-relative bytes A=210B+192+so and A+4 -> two consecutive words, same byte lane
      uint32_t sab;
      if constexpr ((B&1)==0){ constexpr int C=(210*B+192)/4; sab=__byte_perm(wb[i][C+sc_we],wb[i][C+sc_we+1],sc_sel_e); }
      else                   { constexpr int C=(210*B+190)/4; sab=__byte_perm(wb[i][C+sc_wo],wb[i][C+sc_wo+1],sc_sel_o); }
      const uint32_t sh=__byte_perm(sab^0x8080u,0x64646464u,0x5140);
      constexpr int DA=210*B+208;
      const uint32_t dw=wb[i][DA/4];
      const uint32_t dd=((DA&3)==0) ? __byte_perm(dw,0,0x1010) : __byte_perm(dw,0,0x3232);
      const __half2 d2=__hmul2(*(const __half2*)&dd,k1024);
      const __half2 sc2=__hmul2(__hsub2(*(const __half2*)&sh,k1152),d2);
      const __half2 sA=__low2half2(sc2), sB=__high2half2(sc2);
      const uint32_t qa=(vl&0x0F0F0F0F)|((vh<<4)&0x30303030);
      const uint32_t qb=((vl>>4)&0x0F0F0F0F)|(vh&0x30303030);
      const uint32_t h0=__byte_perm(qa,0x64646464u,0x5140), h1=__byte_perm(qa,0x64646464u,0x5342);
      const uint32_t h2=__byte_perm(qb,0x64646464u,0x5140), h3=__byte_perm(qb,0x64646464u,0x5342);
      const __half2 w0=__hmul2(__hsub2(*(const __half2*)&h0,k1056),sA), w1=__hmul2(__hsub2(*(const __half2*)&h1,k1056),sA);
      const __half2 w2=__hmul2(__hsub2(*(const __half2*)&h2,k1056),sB), w3=__hmul2(__hsub2(*(const __half2*)&h3,k1056),sB);
      #pragma unroll
      for(int c=0;c<NC;c++){
        __half2 u = B==0 ? __hmul2(w0,xa[c][0]) : __hfma2(w0,xa[c][0],t[c][i]);
        u=__hfma2(w1,xa[c][1],u); u=__hfma2(w2,xb[c][0],u); t[c][i]=__hfma2(w3,xb[c][1],u);
      }
    }
    v5_blocks<NC,NBLK,B+1>(wb,xw,t,P,ql_w,qh_w,vh_shift,sc_we,sc_wo,sc_sel_e,sc_sel_o);
  }
}
template<int NC>
__global__ void __launch_bounds__(V5_NW*32,1) v5_mv(const uint8_t* __restrict__ W, const __half* __restrict__ XS, const float* __restrict__ S,
                                                   float* __restrict__ Y, int rows, int K){
  constexpr int RPB=V5_NW*V5_RPW, WB=V5_NBF*210, NU=(WB+15+15)/16;
  const int lane=threadIdx.x, wid=threadIdx.y, tid=wid*32+lane;
  const int nb=K/256, nw=(nb+V5_NBF-1)/V5_NBF; const int row0=blockIdx.x*RPB+wid*V5_RPW;
  __shared__ uint4 wst[V5_NW][V5_RPW][NU];
  __shared__ uint4 xst[NC][V5_WIN/8];
  const int iqs=lane, P=128*(iqs/16)+4*(iqs%16);
  const int ql_w=iqs, qh_w=8*(iqs/16)+iqs%8, vh_shift=2*((iqs%16)/8), so=8*(iqs/16)+(iqs%16)/4;
  const int sc_we=so>>2, sc_wo=(so+2)>>2;
  const uint32_t sc_sel_e=(so&3) | ((4+(so&3))<<4), sc_sel_o=((so+2)&3) | ((4+((so+2)&3))<<4);
  float acc[NC][V5_RPW];
  #pragma unroll
  for(int c=0;c<NC;c++) for(int i=0;i<V5_RPW;i++) acc[c][i]=0.f;
  const char* rp[V5_RPW];
  #pragma unroll
  for(int i=0;i<V5_RPW;i++) rp[i]=(const char*)W+(size_t)min(row0+i,rows-1)*nb*210;
  const __half* xw[NC];
  #pragma unroll
  for(int c=0;c<NC;c++) xw[c]=(const __half*)xst[c];
  constexpr int XU=NC*V5_NBF*32; constexpr int XR=(XU+V5_NW*32-1)/(V5_NW*32);
  const uint4* xsrc[XR]; uint4* xdst[XR]; int xj[XR];
  #pragma unroll
  for(int r=0;r<XR;r++){ const int u=r*V5_NW*32+tid; const int c=u/(V5_NBF*32), j=u%(V5_NBF*32);
    xj[r]= u<XU ? j : 1<<30; xsrc[r]=(const uint4*)(XS+(size_t)min(c,NC-1)*K)+j; xdst[r]=&xst[min(c,NC-1)][j]; }
  const float* spc[NC];
  #pragma unroll
  for(int c=0;c<NC;c++) spc[c]=S+c*nw;
  for(int win=0;win<nw;win++){
    const int nblk=min(V5_NBF,nb-win*V5_NBF);
    __syncthreads();
    const uint32_t* wb[V5_RPW];
    #pragma unroll
    for(int i=0;i<V5_RPW;i++){ const char* g=rp[i]; rp[i]+=WB; const int m=(int)((uintptr_t)g&15); // multiple of 4
      wb[i]=(const uint32_t*)wst[wid][i]+(m>>2);
      const uint4* g16=(const uint4*)(g-m); const int nu=(m+nblk*210+15)/16;
      #pragma unroll
      for(int r=0;r<(NU+31)/32;r++){ const int k=r*32+lane; if(k<nu) wst[wid][i][k]=__ldg(g16+k); } }
    #pragma unroll
    for(int r=0;r<XR;r++){ if(xj[r]<nblk*32) *xdst[r]=__ldg(xsrc[r]); xsrc[r]+=V5_WIN/8; }
    __syncthreads();
    __half2 t[NC][V5_RPW];
    if(nblk==V5_NBF) v5_blocks<NC,V5_NBF>(wb,xw,t,P,ql_w,qh_w,vh_shift,sc_we,sc_wo,sc_sel_e,sc_sel_o);
    else             v5_blocks<NC,2>(wb,xw,t,P,ql_w,qh_w,vh_shift,sc_we,sc_wo,sc_sel_e,sc_sel_o); // tail: nb%NBF == 2
    #pragma unroll
    for(int c=0;c<NC;c++){ const float s=__ldg(spc[c]+win);
      #pragma unroll
      for(int i=0;i<V5_RPW;i++) acc[c][i]=fmaf(s,__low2float(t[c][i])+__high2float(t[c][i]),acc[c][i]); }
  }
  #pragma unroll
  for(int c=0;c<NC;c++){
    #pragma unroll
    for(int i=0;i<V5_RPW;i++){ float v=acc[c][i]; for(int o=16;o;o>>=1) v+=__shfl_xor_sync(~0u,v,o);
      if(lane==i && row0+i<rows) Y[(size_t)c*rows+row0+i]=v; } }
}
size_t cand_scratch(int rows,int K,int nc){ return (size_t)nc*K*2+(size_t)nc*((K+V5_WIN-1)/V5_WIN)*4+256; }
void cand_run(const void*W,const float*X,float*Y,void*scr,int rows,int K,int nc,cudaStream_t st){
  __half* XS=(__half*)scr; float* S=(float*)((char*)scr+(size_t)nc*K*2);
  static int mode=getenv("CAND_MODE")?atoi(getenv("CAND_MODE")):0;
  const int nw=(K+V5_WIN-1)/V5_WIN;
  if(mode!=2){ dim3 pb(32,4), pg((nw+3)/4,nc); v5_prep<<<pg,pb,0,st>>>(X,XS,S,K); }
  if(mode==1) return;
  dim3 bd(32,V5_NW); int g=(rows+V5_NW*V5_RPW-1)/(V5_NW*V5_RPW);
  switch(nc){ case 1: v5_mv<1><<<g,bd,0,st>>>((const uint8_t*)W,XS,S,Y,rows,K); break;
    case 2: v5_mv<2><<<g,bd,0,st>>>((const uint8_t*)W,XS,S,Y,rows,K); break;
    case 3: v5_mv<3><<<g,bd,0,st>>>((const uint8_t*)W,XS,S,Y,rows,K); break;
    case 4: v5_mv<4><<<g,bd,0,st>>>((const uint8_t*)W,XS,S,Y,rows,K); break;
    case 5: v5_mv<5><<<g,bd,0,st>>>((const uint8_t*)W,XS,S,Y,rows,K); break; }
}
