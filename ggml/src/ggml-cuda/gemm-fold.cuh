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
