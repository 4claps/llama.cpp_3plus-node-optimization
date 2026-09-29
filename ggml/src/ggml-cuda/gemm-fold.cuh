#pragma once

#include "common.cuh"

// Prefill GEMM for Pascal: fp16 HFMA2 products with short fp16 chains folded into fp32 accumulators.
// Replaces cuBLAS COMPUTE_16F (whole-K fp16 accumulation) for quantized/f16 weights x f32
// activations. Returns false if it does not handle the shape. See gemm-fold.cu.
bool ggml_cuda_gemm_fold_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst);

// True for a matmul the fold path would take but for its small row count: those run in fp32 cuBLAS
// instead (exact, and cheap at that size), not in whole-K fp16.
bool ggml_cuda_gemm_fold_wants_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   const ggml_tensor * dst);

// Matmul pairing: the graph loop calls set_partner(next node) before running a MUL_MAT whose next node
// is a MUL_MAT on the same src1. If this node takes the fold path and the partner qualifies, both run
// as one launch (one activation prescale); take_partner_done() then returns true and the loop skips
// the partner. Every output is computed exactly as in two launches. GGML_CUDA_GEMM_FOLD_PAIR=0: off.
void ggml_cuda_gemm_fold_set_partner(ggml_tensor * next);
bool ggml_cuda_gemm_fold_take_partner_done();

// Weight prefetch: called by the chunked exchange before the compute stream waits for the peer's
// copies. Dequantizes, on the compute stream, the fold weights that followed this exchange last time
// (learned per device); the next fold matmul uses them if the weight pointer matches. Same dequant,
// only earlier: it fills the exchange wait. GGML_CUDA_FOLD_PREFETCH=0: off.
void ggml_cuda_gemm_fold_prefetch(ggml_backend_cuda_context & ctx);
