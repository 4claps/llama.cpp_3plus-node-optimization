// Trivial candidate: fp32 dequant + fp32 dot, one warp per row. Only to validate the harness.
__global__ void k_ref(const block_q6_K* W,const float*X,float*Y,int rows,int K,int nc){
  int r=blockIdx.x*blockDim.y+threadIdx.y; if(r>=rows) return; int nb=K/256; const block_q6_K*br=W+(size_t)r*nb;
  float acc[8]={0};
  for(int i=0;i<nb;i++){ float d=__half2float(br[i].d); const uint8_t*ql=br[i].ql,*qh=br[i].qh; const int8_t*sc=br[i].scales;
    for(int h=0;h<2;h++){ int l=threadIdx.x; int is=l/16;
      int q[4]={((ql[h*64+l]&0xF)|(((qh[h*32+l]>>0)&3)<<4))-32,((ql[h*64+l+32]&0xF)|(((qh[h*32+l]>>2)&3)<<4))-32,
                ((ql[h*64+l]>>4)|(((qh[h*32+l]>>4)&3)<<4))-32,((ql[h*64+l+32]>>4)|(((qh[h*32+l]>>6)&3)<<4))-32};
      for(int j=0;j<4;j++){ float w=d*sc[h*8+is+2*j]*q[j]; int k=i*256+h*128+l+32*j; for(int c=0;c<nc;c++) acc[c]+=w*X[(size_t)c*K+k]; } } }
  for(int c=0;c<nc;c++){ float v=acc[c]; for(int o=16;o;o>>=1) v+=__shfl_xor_sync(~0u,v,o); if(threadIdx.x==0) Y[(size_t)c*rows+r]=v; } }
size_t cand_scratch(int,int,int){return 0;}
void cand_run(const void*W,const float*X,float*Y,void*,int rows,int K,int nc,cudaStream_t s){
  dim3 bd(32,4); k_ref<<<(rows+3)/4,bd,0,s>>>((const block_q6_K*)W,X,Y,rows,K,nc); }
