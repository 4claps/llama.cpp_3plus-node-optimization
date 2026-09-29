// Prefill GEMM for Pascal (P100): fp16 products, fp32 accumulation, at close to fp16 speed.
//
// cuBLAS COMPUTE_16F (what the f16 path used) accumulates each output over the whole of K in fp16:
// NMSE ~1.2e-5 against an fp64 product even with the blocked ALGO6, about half of the f16 path's
// model-level distance from fp32 (OPTLOG attempts 153, 221).
// COMPUTE_32F fixes it but runs at the fp32 FMA rate (-40% prefill).
//
// Here each thread keeps an 8x8 output tile as half2 accumulators (lanes = even/odd k) and every
// GGML_CUDA_GEMM_FOLD_K2 (default 128) k2 steps, i.e. 256 values of K, adds the two lanes (one
// HADD2) and moves the sum into fp32 with an integer half->float conversion: for a half in the high
// lane of a 32-bit word, an arithmetic shift right by 3 and a mask give the fp32 bit pattern of
// value * 2^-112 (the rebias is undone once, in the epilogue). 4 instructions per output per fold,
// where F2F would cost 4x more issue slots. The fp16 chains no longer grow with K: NMSE 1.4e-6
// against fp64 on a real 8704x5120 q6_K slice, cuBLAS COMPUTE_16F 1.2e-5 (OPTLOG attempt 221).
//
// What is left is the f16 rounding of the inputs, which any f16-multiply path has: KLD against an
// all-fp32 run 0.00125, cuBLAS f16 0.00152, all-fp32 with a different summation order 0.0006-0.001.
//
// Activations are prescaled per column by a power of 2 (exact) so their max lands in [8,16): the
// fp16 partial sums stay far from overflow and from the subnormal range.
//
// Outputs are rounded to f16 by default (GGML_CUDA_GEMM_FOLD=2): that keeps the tensor-parallel
// exchange compressed (f16-exact partials) and measured no less accurate than fp32 outputs
// (GGML_CUDA_GEMM_FOLD=1). Matmuls under GGML_CUDA_GEMM_FOLD_MINROWS (1024) rows run in fp32
// cuBLAS instead, since a 128-row tile leaves most SMs idle there. GGML_CUDA_GEMM_FOLD=0 falls back
// to cuBLAS COMPUTE_16F.

#include "gemm-fold.cuh"
#include "convert.cuh"

#include <algorithm>

namespace {

constexpr int BM  = 128;
constexpr int BN  = 128;
constexpr int BK2 = 16;     // k2 steps per smem tile = 32 values of K

__global__ void gemm_fold_prescale(const float * __restrict__ X, const int64_t s1, half * __restrict__ X16,
                                   float * __restrict__ cs, const int K, const int xexp) {
    const int n = blockIdx.x;
    const float * x = X + n*s1;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        m = fmaxf(m, fabsf(x[k]));
    }
    __shared__ float sm[32];
    m = warp_reduce_max(m);
    if ((threadIdx.x & 31) == 0) {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < blockDim.x/32 ? sm[threadIdx.x] : 0.0f;
        m = warp_reduce_max(m);
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

static __device__ __forceinline__ float gemm_fold_h(const half2 p) {
    const half2    s = __hadd2(p, __lowhigh2highlow(p));  // high lane = lo + hi
    const int32_t  v = ((int32_t) *(const uint32_t *) &s) >> 3;
    return __int_as_float(v & 0x8FFFE000);                 // = (lo + hi) * 2^-112; subnormal halves flush
}

template <int fold_k2, bool out16>
__global__ void __launch_bounds__(256, 1) gemm_fold_kernel(
        const half * __restrict__ W, const half * __restrict__ X, const float * __restrict__ cs,
        float * __restrict__ Y, const int M, const int N, const int K, const int64_t sy, half * __restrict__ R) {
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
            if (out16 && R != nullptr) {
                // the tensor-parallel peer's landing buffer, over P2P: the same f16 values the
                // compressed peer copy would send (v is f16-exact here)
                half2 r[2] = {__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w)};
                *(uint2 *) (R + n*sy + m) = *(const uint2 *) r;
            }
        }
    }
}

// The same products, chains and folds as gemm_fold_kernel<128, true>, with cheaper bookkeeping:
// load/store offsets computed once, the tile loop unrolled by two (compile-time buffer index), the
// fold as HADD2.F32 with the 2^-112 moved into the column scale (a power of two, so every rounding
// is unchanged), and blocks grouped by RASTER weight row-blocks for L2 reuse. Needs K % 64 == 0.
// Harness 8704x5120 N=2048: 14.57 -> 13.68 ms, 0 of 17.8M outputs differ.
constexpr int GEMM_FOLD_RASTER = 4;

__global__ void __launch_bounds__(256, 1) gemm_fold_kernel_u2(
        const half * __restrict__ W, const half * __restrict__ X, const float * __restrict__ cs,
        float * __restrict__ Y, const int M, const int N, const int K, const int64_t sy) {
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BN];

    const int t   = threadIdx.x;
    const int tx  = t & 15;
    const int ty  = t >> 4;
    const int gm  = gridDim.x;
    const int gn  = gridDim.y;
    const int lin = blockIdx.x + blockIdx.y*gm;
    const int grp = lin / (GEMM_FOLD_RASTER*gn);
    const int gsz = min(GEMM_FOLD_RASTER, gm - grp*GEMM_FOLD_RASTER);
    const int m0  = (grp*GEMM_FOLD_RASTER + (lin % (GEMM_FOLD_RASTER*gn)) % gsz)*BM;
    const int n0  = ((lin % (GEMM_FOLD_RASTER*gn)) / gsz)*BN;

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

    const uint4 * pa[2];
    const uint4 * pb[2];
    bool va[2], vb[2];
    int  so[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int l = t + 256*i, r = l >> 2, c = l & 3;
        va[i] = m0 + r < M;
        vb[i] = n0 + r < N;
        pa[i] = (const uint4 *) (W + (int64_t) (va[i] ? m0 + r : 0)*K + c*8);
        pb[i] = (const uint4 *) (X + (int64_t) (vb[i] ? n0 + r : 0)*K + c*8);
        so[i] = (c*4)*BM + (r ^ (c << 3));   // same XOR swizzle as gemm_fold_kernel
    }
    int ao[4], bo[4];
#pragma unroll
    for (int g = 0; g < 4; g++) {
        ao[g] = (ty*4) ^ (g << 3);
        bo[g] = (tx*4) ^ (g << 3);
    }

    uint4 ra[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    uint4 rb[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    auto gload = [&](const int it) {   // tile it = K offset it*32 halves = it*4 uint4
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (va[i]) ra[i] = pa[i][it*4];
            if (vb[i]) rb[i] = pb[i][it*4];
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            uint32_t * a = &As[buf][0][0] + so[i];
            uint32_t * b = &Bs[buf][0][0] + so[i];
            a[0] = ra[i].x; a[BM] = ra[i].y; a[2*BM] = ra[i].z; a[3*BM] = ra[i].w;
            b[0] = rb[i].x; b[BN] = rb[i].y; b[2*BN] = rb[i].z; b[3*BN] = rb[i].w;
        }
    };
    auto tile = [&](const int buf, const bool restart) {
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const uint4 a0 = *(const uint4 *) &As[buf][k2][ao[k2 >> 2]];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][ao[k2 >> 2] + 64];
            const uint4 b0 = *(const uint4 *) &Bs[buf][k2][bo[k2 >> 2]];
            const uint4 b1 = *(const uint4 *) &Bs[buf][k2][bo[k2 >> 2] + 64];
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
    };
    auto fold = [&]() {
#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[i][j] += __half2float(__hadd(__low2half(h[i][j]), __high2half(h[i][j])));
            }
        }
    };

    constexpr int TPF = 128/BK2;       // tiles per fold
    const int nt = K / (2*BK2);        // even: K % 64 == 0
    gload(0);
    sstore(0);
    __syncthreads();
    for (int it = 0; it < nt; it += 2) {
        gload(it + 1);
        tile(0, it % TPF == 0);
        sstore(1);
        __syncthreads();
        if (it + 2 < nt) {
            gload(it + 2);
        }
        tile(1, false);
        if ((it + 2) % TPF == 0 || it + 2 >= nt) {
            fold();
        }
        if (it + 2 < nt) {
            sstore(0);
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = n0 + (j < 4 ? tx*4 + j : 64 + tx*4 + j - 4);
        if (n >= N) {
            continue;
        }
        const float s = cs[n] * 0x1p-112f;
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih*64 + ty*4;
            if (m >= M) {
                continue;
            }
            float4 v = make_float4(acc[ih*4 + 0][j]*s, acc[ih*4 + 1][j]*s, acc[ih*4 + 2][j]*s, acc[ih*4 + 3][j]*s);
            v.x = __half2float(__float2half(v.x)); v.y = __half2float(__float2half(v.y));
            v.z = __half2float(__float2half(v.z)); v.w = __half2float(__float2half(v.w));
            *(float4 *) (Y + n*sy + m) = v;
        }
    }
}

} // namespace

static int ggml_cuda_gemm_fold_env(const char * name, int def) {
    const char * s = getenv(name);
    return s ? atoi(s) : def;
}

static int ggml_cuda_gemm_fold_mode() {
    // 0: off (cuBLAS), 1: on, fp32 outputs, 2: on, outputs rounded to f16 (keeps the f16 peer exchange)
    static const int v = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD", 2);
    return v;
}

static int ggml_cuda_gemm_fold_k2() {
    // half2 steps per fp16 chain (16, 32, 64, 128): 128 -> a fold every 256 values of K
    static const int v = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_K2", 128);
    return v;
}

static int ggml_cuda_gemm_fold_xexp() {
    // per-column prescale puts max|x| in [2^xexp, 2^(xexp+1)); -1 disables the prescale
    static const int v = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_XEXP", 3);
    return v;
}

static int ggml_cuda_gemm_fold_min_rows() {
    // below this many rows a 128-row tile leaves most SMs idle; those (small) matmuls run in fp32 cuBLAS
    static const int v = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_MINROWS", 1024);
    return v;
}

static bool ggml_cuda_gemm_fold_eligible(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                         const ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return ggml_cuda_gemm_fold_mode() != 0 && cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_VOLTA && fast_fp16_available(cc) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
}

bool ggml_cuda_gemm_fold_wants_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   const ggml_tensor * dst) {
    return ggml_cuda_gemm_fold_eligible(ctx, src0, src1, dst) && src0->ne[1] < ggml_cuda_gemm_fold_min_rows();
}

bool ggml_cuda_gemm_fold_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst) {
    const int mode = ggml_cuda_gemm_fold_mode();
    if (!ggml_cuda_gemm_fold_eligible(ctx, src0, src1, dst) || src0->ne[1] < ggml_cuda_gemm_fold_min_rows()) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        K % 32 != 0 || M % 4 != 0 || src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(dst) || M > INT_MAX || N > 65535*BN) {
        return false;
    }
    const to_fp16_cuda_t to_fp16 = src0->type == GGML_TYPE_F16 ? nullptr : ggml_get_to_fp16_cuda(src0->type);
    if (src0->type != GGML_TYPE_F16 && to_fp16 == nullptr) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    const half * W16 = (const half *) src0->data;
    ggml_cuda_pool_alloc<half> w_alloc(ctx.pool());
    if (to_fp16) {
        w_alloc.alloc(ggml_nelements(src0));
        to_fp16(src0->data, w_alloc.get(), ggml_nelements(src0), stream);
        W16 = w_alloc.get();
    }

    ggml_cuda_pool_alloc<half>  x_alloc(ctx.pool(), N*K);
    ggml_cuda_pool_alloc<float> s_alloc(ctx.pool(), N);
    gemm_fold_prescale<<<N, 256, 0, stream>>>((const float *) src1->data, src1->nb[1]/sizeof(float),
                                              x_alloc.get(), s_alloc.get(), K, ggml_cuda_gemm_fold_xexp());

    const dim3 grid((M + BM - 1)/BM, (N + BN - 1)/BN);
    const int64_t sy = dst->nb[1]/sizeof(float);
    const half  * X16 = x_alloc.get();
    const float * cs  = s_alloc.get();
    float       * Y   = (float *) dst->data;
    const bool    o16 = mode == 2;

    // exchange source: token chunks with an event after each (see xchg in common.cuh)
    static const int xchg_chunks = std::max(1, std::min(8, ggml_cuda_gemm_fold_env("GGML_CUDA_XCHG_CHUNKS", 4)));
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    // direct mode (GGML_CUDA_XCHG_DIRECT=1, off): also write the f16 outputs straight into the peer's landing
    // buffer over P2P. Measured 253 t/s against 411 for the chunked copies: the epilogue's scattered 8-byte
    // stores make poor PCIe transactions across the two CPU root ports (OPTLOG 237).
    static const bool xchg_direct = ggml_cuda_gemm_fold_env("GGML_CUDA_XCHG_DIRECT", 0) != 0;
    ggml_backend_cuda_context * peer = ctx.xchg_peer;
    // GGML_CUDA_GEMM_FOLD_U2=0 falls back to gemm_fold_kernel (bit-identical, slower)
    static const bool u2_env = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_U2", 1) != 0;
    const bool u2 = u2_env && o16 && K % 64 == 0 && ggml_cuda_gemm_fold_k2() == 128;
    if (ctx.xchg_want && xchg_direct && peer != nullptr && ctx.peer_f16_ok == 1 && N >= 512 && o16 &&
            ggml_cuda_gemm_fold_k2() == 128 && capturing == cudaStreamCaptureStatusNone && ggml_is_contiguous(dst)) {
        auto & xc = ctx.xchg;
        half * R = (half *) peer->peer_stage_get(ggml_backend_cuda_context::PEER_STAGE_IN, (size_t) M*N*sizeof(half));
        if (peer->peer_stage_free == nullptr) {
            ggml_cuda_set_device(peer->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&peer->peer_stage_free, cudaEventDisableTiming));
        }
        ggml_cuda_set_device(ctx.device);
        if (xc.ev[0] == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&xc.ev[0], cudaEventDisableTiming));
        }
        // the peer must have consumed its previous delivery before this kernel overwrites it
        CUDA_CHECK(cudaStreamWaitEvent(stream, peer->peer_stage_free, 0));
        gemm_fold_kernel<128, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, R);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(xc.ev[0], stream));
        xc.n = 1;
        xc.rows = M;
        xc.col[0] = 0;
        xc.col[1] = N;
        xc.direct = true;
        xc.data = dst->data;
        return true;
    }

    if (ctx.xchg_want && xchg_chunks > 1 && N >= 512 && o16 && ggml_cuda_gemm_fold_k2() == 128 &&
            capturing == cudaStreamCaptureStatusNone && ggml_is_contiguous(dst)) {
        const int64_t nb   = (N + BN - 1)/BN;
        const int     nch  = (int) std::min<int64_t>(xchg_chunks, nb);
        auto & xc = ctx.xchg;
        xc.n = nch;
        xc.rows = M;
        for (int c = 0; c <= nch; c++) {
            xc.col[c] = std::min<int64_t>(N, (nb*c/nch)*BN);
        }
        for (int c = 0; c < nch; c++) {
            if (xc.ev[c] == nullptr) {
                CUDA_CHECK(cudaEventCreateWithFlags(&xc.ev[c], cudaEventDisableTiming));
            }
            const int64_t n0 = xc.col[c], nc = xc.col[c + 1] - n0;
            const dim3 gc((M + BM - 1)/BM, (nc + BN - 1)/BN);
            if (u2) {
                gemm_fold_kernel_u2<<<gc, 256, 0, stream>>>(W16, X16 + n0*K, cs + n0, Y + n0*sy, M, (int) nc, K, sy);
            } else {
                gemm_fold_kernel<128, true><<<gc, 256, 0, stream>>>(W16, X16 + n0*K, cs + n0, Y + n0*sy, M, (int) nc, K, sy, nullptr);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(xc.ev[c], stream));
        }
        xc.direct = false;
        xc.data = dst->data;
        return true;
    }

    if (u2) {
        gemm_fold_kernel_u2<<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    switch (ggml_cuda_gemm_fold_k2()) {
        case 16: o16 ? gemm_fold_kernel<16, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr)
                     : gemm_fold_kernel<16, false><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr); break;
        default:  o16 ? gemm_fold_kernel<128, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr)
                      : gemm_fold_kernel<128, false><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr); break;
        case 64: o16 ? gemm_fold_kernel<64, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr)
                     : gemm_fold_kernel<64, false><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr); break;
        case 32: o16 ? gemm_fold_kernel<32, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr)
                     : gemm_fold_kernel<32, false><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
