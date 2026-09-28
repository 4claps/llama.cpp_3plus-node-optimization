# Goal: 500+ t/s prefill at 0 context, 300+ t/s at 260k (2x P100, power caps unchanged)

Branch `goal/prefill300`. Running log; newest at the bottom. Resume from the last entry.

## Baseline (2026-09-28, cold cards, 175 W caps)

| test | t/s |
|---|---|
| pp2048 d0, -ub 2048, fold GEMM (shipped) | 382-392 |
| same, GGML_CUDA_GEMM_FOLD=0 (stock cuBLAS fp16, less accurate: not allowed) | 414 |
| pp4096 d0, -ub 4096 | 378 |
| 260k, ~30k-token chunk (depth sweep 09-27, -ub 1024) | 106 |

## Where the time goes at 0 context (GGML_CUDA_OP_PROFILE=1, per card)

- weight matmuls (fold) 79.6%, at ~12.6 TFLOPS (66% of 19.05 peak at 1328 MHz)
- fused GATED_DELTA_NET 7.5%
- fused RMS_NORM n=24 (the delta-net gated output norm) 6.2%: 6 ms per call for ~25 MB -> broken shape
- FLASH_ATTN_EXT 2.6%, the rest ~4%
- bench wall ~10% above summed op time: host gaps between ~257 subgraphs per pass

## Arithmetic

Per token per card: weights ~24.8 GFLOP; attention at depth d: 16 layers x 12 heads x 256 x 4 x d
= 51.5 GFLOP at 262144. Peak 19.05 TFLOPS (fp16x2 at 1328 MHz; ~17 at the 1189 MHz sustained clock).
Dense exact attention caps 260k at ~240 t/s even at 100% of peak. 300 needs attention work cut to
~15% (block-sparse) AND the non-attention part near its 0-context speed of ~450-500 t/s.

## Plan

A. 0 context, exact: RMS_NORM n=24 fix; fold GEMM toward cuBLAS speed; host gaps; chunked delta-net.
B. 260k: block-sparse prefill attention. Step 1: measure this model's attention concentration at
   depth (GGML_CUDA_FA_SPARSITY=1 over slot snapshots in /mnt/fast/p100-scratch/slots).
   Accuracy gate: KLD/PPL at long context vs dense, plus retrieval at depth.

## 2026-09-28: attention concentration at depth (GGML_CUDA_FA_SPARSITY=1, ~1.5k new tokens on slot snapshots)

| depth | 128-key tiles for 99% of a query's mass | 99.9% | 99.99% | 128x128 blocks kept at 99% |
|---|---|---|---|---|
| 64k | 0.48-0.50 | 0.74-0.76 | 0.89-0.90 | 0.98-0.99 |
| 128k | 0.44-0.47 | 0.71-0.73 | 0.87-0.89 | 0.98 |
| 260k | 0.50 | 0.76 | 0.90 | 0.98 |

Diffuse at every depth (gated attention: no attention sink). Block-sparse prefill (MInference, FlexPrefill,
XAttention, 128x128 granularity) would skip ~2% here: dead end for this model.

Research: arXiv 2606.07703 (Oracle-guided sparse prefill in hybrid models) on Qwen3.5-27B at 128K:
per-query top-2048 tokens (98.4% sparse) kept RULER within ~0.7 pt of dense; sharing the selection
across 64-query blocks collapses retrieval tasks. UniPrefill on Qwen3-Next: 1.68x at 128K, -0.7 pt RULER.
Loki (NeurIPS 2024): low-dim (PCA) key scoring ranks keys well enough to pick top-k.
Candidate design for 300+ at 260k: low-dim scoring (r = 32-64 of 256) + per-query top-k + exact
attention over the chosen keys. Next: GGML_CUDA_FA_ORACLE_DELTA (exact scores, drop logits more than
delta below the query max) to measure the true output error on this model per sparsity level.

## Correction: 0-context breakdown

"fused:RMS_NORM n=24" is the gated-norm fusion group, which runs the z-projection matmul (5120x3072)
inside it: the norm is not slow. Real split at pp2048 d0: matmuls ~86% (~12.6 TFLOPS), GATED_DELTA_NET
7.5% (recurrent kernel at ~0.44 TFLOPS: latency-bound), FA 2.6%, rest ~4%. Wall vs op-sum gap (~10%)
matches 128 cross-GPU exchanges per ubatch (21 MB each at ~7.1 GB/s = ~0.37 s per 2048 tokens,
attempt 184), which run between subgraphs and are not overlapped with compute.

Exact levers for 0 context: (1) overlap the TP exchange with compute (micro-batch the ubatch),
~-7%; (2) chunked delta-net (a graph path exists: build_delta_net_chunking; cparams.fused_gdn_ch picks
the recurrent CUDA op), up to ~-5%; (3) fold GEMM efficiency (12.6 -> 14+ TFLOPS), ~-10%.

## 0-context GEMM lever

Under prefill both cards sit at the 175 W cap at ~1252 MHz (peak there: 17.95 TFLOPS). The fold GEMM
runs ~12.6 TFLOPS = ~70%; cuBLAS hgemm reached ~84%. gemm_fold_kernel is __launch_bounds__(256, 1):
one CTA per SM, so smem-store/sync/global-load phases are exposed. fa_fold_qk2 fixed the same issue
with 2 CTAs/SM (+20%). 500 at 0 context needs the matmuls at ~81% plus exchange/GDN savings.
Candidate: a 2-CTA/SM fold GEMM (smaller per-thread tile or fp32 partials staged differently).
