// Warp-specialised variant of the fold GEMM (fold_kernel.cuh): 8 compute warps + 2 loader warps and
// a 3-stage shared-memory ring. Loaders move tile t (global -> registers -> smem) while the compute
// warps work on earlier tiles; the groups hand stages over with named barriers (bar.arrive /
// bar.sync), so no compute warp ever waits on a global load and there is no CTA-wide barrier per
// tile. The smem layout (swizzle included), the per-thread output tile and every fp16 chain / fp32
// fold are unchanged, so each output is computed by the same operations in the same order.
#include <cuda_fp16.h>
#include <cstdint>
constexpr int WS_BM = 128;
constexpr int WS_BN = 128;
constexpr int WS_BK2 = 16;
constexpr int WS_NS = 3;         // ring stages
constexpr int WS_NC = 256;       // compute threads
constexpr int WS_NL = 64;        // loader threads
constexpr int WS_NT = WS_NC + WS_NL;

static __device__ __forceinline__ float ws_fold_h(const half2 p) {
    const half2    s = __hadd2(p, __lowhigh2highlow(p));
    const int32_t  v = ((int32_t) *(const uint32_t *) &s) >> 3;
    return __int_as_float(v & 0x8FFFE000);
}
static __device__ __forceinline__ void ws_bar_sync(int id)   { asm volatile("bar.sync %0, %1;"   :: "r"(id), "r"(WS_NT) : "memory"); }
static __device__ __forceinline__ void ws_bar_arrive(int id) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(WS_NT) : "memory"); }
// barrier ids: FULL[s] = 1 + s (loaders arrive, compute syncs), EMPTY[s] = 1 + WS_NS + s

extern "C" __global__ void __launch_bounds__(WS_NT, 1) fold_gemm_ws(
        const half * __restrict__ W, const half * __restrict__ X, const float * __restrict__ cs,
        float * __restrict__ Y, const int M, const int N, const int K, const int64_t sy) {
    __shared__ __align__(16) uint32_t As[WS_NS][WS_BK2][WS_BM];
    __shared__ __align__(16) uint32_t Bs[WS_NS][WS_BK2][WS_BN];

    const int t  = threadIdx.x;
    const int m0 = blockIdx.x*WS_BM;
    const int n0 = blockIdx.y*WS_BN;
    const int nt = K / (2*WS_BK2);

    if (t >= WS_NC) {
        // ---- loaders: tile it into stage it % NS, 8 A and 8 B uint4 per thread
        const int lt = t - WS_NC;
        for (int it = 0; it < nt; it++) {
            const int s = it % WS_NS;
            if (it >= WS_NS) {
                ws_bar_sync(1 + WS_NS + s);          // compute is done with this stage
            }
            const int k0 = it*2*WS_BK2;
            uint4 ra[8], rb[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int l = lt + WS_NL*i;
                const int r = l >> 2;
                const int c = l & 3;
                ra[i] = m0 + r < M ? *(const uint4 *) (W + (int64_t) (m0 + r)*K + k0 + c*8) : make_uint4(0, 0, 0, 0);
                rb[i] = n0 + r < N ? *(const uint4 *) (X + (int64_t) (n0 + r)*K + k0 + c*8) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int l  = lt + WS_NL*i;
                const int r  = l >> 2;
                const int c  = l & 3;
                const int rs = r ^ (c << 3);
                As[s][c*4 + 0][rs] = ra[i].x; As[s][c*4 + 1][rs] = ra[i].y;
                As[s][c*4 + 2][rs] = ra[i].z; As[s][c*4 + 3][rs] = ra[i].w;
                Bs[s][c*4 + 0][rs] = rb[i].x; Bs[s][c*4 + 1][rs] = rb[i].y;
                Bs[s][c*4 + 2][rs] = rb[i].z; Bs[s][c*4 + 3][rs] = rb[i].w;
            }
            ws_bar_arrive(1 + s);                     // stage s is full
        }
        return;
    }

    // ---- compute: identical to fold_kernel.cuh's loop apart from where the tile comes from
    const int tx = t & 15;
    const int ty = t >> 4;
    float acc[8][8];
    half2 h[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
            h[i][j]   = make_half2(0.0f, 0.0f);
        }
    }
    for (int it = 0; it < nt; it++) {
        const int s = it % WS_NS;
        ws_bar_sync(1 + s);                           // wait for the loaders to fill stage s
        const bool restart = (it*WS_BK2) % 128 == 0;
#pragma unroll
        for (int k2 = 0; k2 < WS_BK2; k2++) {
            const int   sw = (k2 >> 2) << 3;
            const uint4 a0 = *(const uint4 *) &As[s][k2][(ty*4) ^ sw];
            const uint4 a1 = *(const uint4 *) &As[s][k2][(64 + ty*4) ^ sw];
            const uint4 b0 = *(const uint4 *) &Bs[s][k2][(tx*4) ^ sw];
            const uint4 b1 = *(const uint4 *) &Bs[s][k2][(64 + tx*4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    h[i][j] = k2 == 0 && restart ? __hmul2(ai, bj) : __hfma2(ai, bj, h[i][j]);
                }
            }
        }
        if (it + WS_NS < nt) {
            ws_bar_arrive(1 + WS_NS + s);             // stage s may be refilled
        }
        if (((it + 1)*WS_BK2) % 128 == 0 || it + 1 == nt) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[i][j] += ws_fold_h(h[i][j]);
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = n0 + (j < 4 ? tx*4 + j : 64 + tx*4 + j - 4);
        if (n >= N) {
            continue;
        }
        const float sc = cs[n];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih*64 + ty*4;
            if (m >= M) {
                continue;
            }
            float4 v = make_float4(acc[ih*4 + 0][j]*sc, acc[ih*4 + 1][j]*sc, acc[ih*4 + 2][j]*sc, acc[ih*4 + 3][j]*sc);
            v.x = __half2float(__float2half(v.x)); v.y = __half2float(__float2half(v.y));
            v.z = __half2float(__float2half(v.z)); v.w = __half2float(__float2half(v.w));
            *(float4 *) (Y + n*sy + m) = v;
        }
    }
}
