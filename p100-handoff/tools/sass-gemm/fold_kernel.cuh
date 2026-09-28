// Exact copy of ggml-cuda/gemm-fold.cu's gemm_fold_kernel<128, true> (the shipped prefill GEMM),
// as a plain extern "C" kernel so it can be compiled to a cubin, edited at the SASS level and
// loaded back. Keep in sync with gemm-fold.cu.
#include <cuda_fp16.h>
#include <cstdint>
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK2 = 16;
constexpr int fold_k2 = 128;
constexpr bool out16 = true;
static __device__ __forceinline__ float gemm_fold_h(const half2 p) {
    const half2    s = __hadd2(p, __lowhigh2highlow(p));  // high lane = lo + hi
    const int32_t  v = ((int32_t) *(const uint32_t *) &s) >> 3;
    return __int_as_float(v & 0x8FFFE000);                 // = (lo + hi) * 2^-112; subnormal halves flush
}

extern "C" __global__ void __launch_bounds__(256, 1) fold_gemm(
        const half * __restrict__ W, const half * __restrict__ X, const float * __restrict__ cs,
        float * __restrict__ Y, const int M, const int N, const int K, const int64_t sy) {
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BN];

    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const int m0 = blockIdx.x*BM;
    const int n0 = blockIdx.y*BN;

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

    uint4 ra[2];
    uint4 rb[2];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            const int r = l >> 2;
            const int c = l & 3;
            ra[i] = m0 + r < M ? *(const uint4 *) (W + (int64_t) (m0 + r)*K + k0 + c*8) : make_uint4(0, 0, 0, 0);
            rb[i] = n0 + r < N ? *(const uint4 *) (X + (int64_t) (n0 + r)*K + k0 + c*8) : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            const int r = l >> 2;
            const int c = l & 3;
            // XOR swizzle: the 4 threads of a row write 4 different k2 rows of the tile, which
            // without it land in the same bank (4-way conflict); reads stay 16-byte vectors
            const int rs = r ^ (c << 3);
            As[buf][c*4 + 0][rs] = ra[i].x; As[buf][c*4 + 1][rs] = ra[i].y;
            As[buf][c*4 + 2][rs] = ra[i].z; As[buf][c*4 + 3][rs] = ra[i].w;
            Bs[buf][c*4 + 0][rs] = rb[i].x; Bs[buf][c*4 + 1][rs] = rb[i].y;
            Bs[buf][c*4 + 2][rs] = rb[i].z; Bs[buf][c*4 + 3][rs] = rb[i].w;
        }
    };

    const int nt = K / (2*BK2);
    gload(0);
    sstore(0);
    __syncthreads();

    for (int it = 0; it < nt; it++) {
        const int buf = it & 1;
        if (it + 1 < nt) {
            gload((it + 1)*2*BK2);
        }
        // the first product of a chain starts it (HMUL2) instead of zeroing 64 registers
        const bool restart = (it*BK2) % fold_k2 == 0;
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int   sw = (k2 >> 2) << 3;
            const uint4 a0 = *(const uint4 *) &As[buf][k2][(ty*4) ^ sw];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][(64 + ty*4) ^ sw];
            const uint4 b0 = *(const uint4 *) &Bs[buf][k2][(tx*4) ^ sw];
            const uint4 b1 = *(const uint4 *) &Bs[buf][k2][(64 + tx*4) ^ sw];
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
        if (((it + 1)*BK2) % fold_k2 == 0 || it + 1 == nt) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[i][j] += gemm_fold_h(h[i][j]);
                }
            }
        }
        if (it + 1 < nt) {
            sstore(buf ^ 1);
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = n0 + (j < 4 ? tx*4 + j : 64 + tx*4 + j - 4);
        if (n >= N) {
            continue;
        }
        const float s = cs[n];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih*64 + ty*4;
            if (m >= M) {
                continue;
            }
            float4 v = make_float4(acc[ih*4 + 0][j]*s, acc[ih*4 + 1][j]*s, acc[ih*4 + 2][j]*s, acc[ih*4 + 3][j]*s);
            if (out16) {
                v.x = __half2float(__float2half(v.x)); v.y = __half2float(__float2half(v.y));
                v.z = __half2float(__float2half(v.z)); v.w = __half2float(__float2half(v.w));
            }
            *(float4 *) (Y + n*sy + m) = v;
        }
    }
}

