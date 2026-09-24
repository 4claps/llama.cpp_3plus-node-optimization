// Fast loop for a candidate q6_K x f32 multi-column matvec (the MTP verify shape).
//   nvcc -O3 -arch=sm_60 -DCAND='"v1.cuh"' -o h harness.cu && ./h gate_8704x5120.q6k 8704 5120 5
// Reports NMSE vs a double reference for the candidate AND for today's q8_1 path (CPU model),
// and the candidate's GPU time (median/min of 200 runs, x prep included).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_fp16.h>
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; __half d; };
static_assert(sizeof(block_q6_K)==210,"q6_K size");
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#include CAND   // must define: size_t cand_scratch(int rows,int K,int nc); void cand_run(const void*W,const float*X,float*Y,void*scratch,int rows,int K,int nc,cudaStream_t)
static void deq_row(const block_q6_K* b, int nb, double* y){
  for(int i=0;i<nb;i++){ double d=__half2float(b[i].d); const uint8_t*ql=b[i].ql,*qh=b[i].qh; const int8_t*sc=b[i].scales;
    for(int n=0;n<256;n+=128){ for(int l=0;l<32;l++){ int is=l/16;
        int q1=((ql[l]&0xF)|(((qh[l]>>0)&3)<<4))-32, q2=((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
        int q3=((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32,   q4=((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
        y[n+l]=d*sc[is]*q1; y[n+l+32]=d*sc[is+2]*q2; y[n+l+64]=d*sc[is+4]*q3; y[n+l+96]=d*sc[is+6]*q4; }
      ql+=64; qh+=32; sc+=8; } y+=256; } }
int main(int argc,char**argv){
  if(argc<5){printf("usage: h W.q6k rows K ncols [outlier_frac=0.002] [reps=200]\n");return 1;}
  const char*fn=argv[1]; int rows=atoi(argv[2]),K=atoi(argv[3]),nc=atoi(argv[4]); double of=argc>5?atof(argv[5]):0.002; int reps=argc>6?atoi(argv[6]):200;
  int nb=K/256; size_t wbytes=(size_t)rows*nb*210; std::vector<uint8_t> W(wbytes);
  FILE*f=fopen(fn,"rb"); if(!f||fread(W.data(),1,wbytes,f)!=wbytes){printf("read %s failed\n",fn);return 1;} fclose(f);
  std::mt19937 rng(42); std::normal_distribution<float> N(0,1); std::uniform_real_distribution<float> U(0,1);
  std::vector<float> X((size_t)nc*K); for(auto&v:X){ v=N(rng); if(U(rng)<of) v*=60; }
  // references
  std::vector<double> ref((size_t)nc*rows), today((size_t)nc*rows), wr(K);
  std::vector<int8_t> xq((size_t)nc*K); std::vector<float> xd((size_t)nc*K/32);
  for(int c=0;c<nc;c++) for(int b=0;b<K/32;b++){ float am=0; for(int j=0;j<32;j++) am=std::max(am,fabsf(X[(size_t)c*K+b*32+j]));
      float d=am/127; xd[(size_t)c*K/32+b]=d; for(int j=0;j<32;j++) xq[(size_t)c*K+b*32+j]=(int8_t)roundf(d?X[(size_t)c*K+b*32+j]/d:0); }
  for(int r=0;r<rows;r++){ const block_q6_K* br=(const block_q6_K*)(W.data()+(size_t)r*nb*210); deq_row(br,nb,wr.data());
    for(int c=0;c<nc;c++){ double s=0; for(int k=0;k<K;k++) s+=wr[k]*(double)X[(size_t)c*K+k]; ref[(size_t)c*rows+r]=s; }
    // today's path: exact int per 16-group, q8_1 scale per 32, fp32 scaling
    for(int c=0;c<nc;c++){ float s=0; for(int i=0;i<nb;i++){ float d=__half2float(br[i].d); float si=0;
        for(int g=0;g<16;g++){ int base=i*256+g*16; int isum=0; for(int j=0;j<16;j++){ int k=base+j; int q=(int)lround(wr[k]/(d*br[i].scales[ (k%256)/16 ] ? d*br[i].scales[(k%256)/16] : 1)); isum+=q*xq[(size_t)c*K+k]; }
          si+= (float)br[i].scales[g] * xd[(size_t)c*K/32+base/32] * (float)isum; }
        s+=d*si; } today[(size_t)c*rows+r]=s; } }
  // GPU
  void *dW,*dS; float *dX,*dY; size_t sc=cand_scratch(rows,K,nc);
  CK(cudaMalloc(&dW,wbytes)); CK(cudaMalloc(&dX,X.size()*4)); CK(cudaMalloc(&dY,(size_t)nc*rows*4)); CK(cudaMalloc(&dS,sc?sc:16));
  CK(cudaMemcpy(dW,W.data(),wbytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));
  cudaStream_t st; CK(cudaStreamCreate(&st)); CK(cudaMemset(dY,0xff,(size_t)nc*rows*4));
  cand_run(dW,dX,dY,dS,rows,K,nc,st); CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
  std::vector<float> Y((size_t)nc*rows); CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
  double e=0,et=0,r2=0,mx=0; int bad=0;
  for(size_t i=0;i<Y.size();i++){ if(!std::isfinite(Y[i])) bad++; double d=Y[i]-ref[i]; e+=d*d; et+=(today[i]-ref[i])*(today[i]-ref[i]); r2+=ref[i]*ref[i]; mx=std::max(mx,fabs(d)); }
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); std::vector<float> t(reps);
  for(int i=0;i<20;i++) cand_run(dW,dX,dY,dS,rows,K,nc,st);
  for(int i=0;i<reps;i++){ cudaEventRecord(a,st); cand_run(dW,dX,dY,dS,rows,K,nc,st); cudaEventRecord(b,st); cudaEventSynchronize(b); cudaEventElapsedTime(&t[i],a,b); }
  std::sort(t.begin(),t.end());
  printf("%s %dx%d n=%d | NMSE cand %.2e  today %.2e  (x%.0f better) | nonfinite %d | us median %.1f  min %.1f\n",
    fn,rows,K,nc,e/r2,et/r2,(et/r2)/(e/r2+1e-30),bad,t[reps/2]*1e3,t[0]*1e3);
  return bad||!(e/r2<et/r2);
}
