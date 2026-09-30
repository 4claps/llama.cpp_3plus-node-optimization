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

## 2026-09-28: user rules out any approximation (math must not change)

Oracle/approximate-attention runs stopped. Exact analysis: at 260k, per token per card, attention
~49 GFLOP (16 layers x 12 heads x 1024 FLOP per query-key pair x ~251k keys) + weights ~25 GFLOP
= ~74 GFLOP. 300 t/s needs ~22 TFLOPS per card; P100 peak is 19.05 (1328 MHz), ~18 at the cap.
Exact ideas examined: q4_0 lookup tables (smem 32/clk vs 128 fp16 FMA/clk: slower), Strassen/
Winograd (fp16 operand sums round: worse math), packed small-int products (no room on Pascal),
key-range attention split to balance the hotter GPU1 (~+4%, exact), CPU co-compute (~+3%),
power-efficient kernels for clock headroom (<= +6%). Exact ceiling ~150-170 t/s at 260k.
4x P100 would make 300 exact arithmetically possible (~11 TFLOPS/card needed).

## 2026-09-28: SASS-level GEMM work (exact), p100-handoff/tools/sass-gemm/

Toolchain: CuAssembler (sm_60 supported) in /mnt/fast/p100-scratch/CuAssembler, venv at
/mnt/fast/p100-scratch/venv. fold_kernel.cuh = exact copy of gemm_fold_kernel<128,true>; cand_fold.cuh
plugs it into gemm-harness and loads an edited cubin from $CUBIN, checking bitwise equality vs the
compiled kernel. Round trip cubin -> cuasm -> cubin: identical SASS.

Finding: 424 register-bank conflicts per tile-loop pass among the 1160 FMA-type instructions (42% of
HFMA2). bankfix.py (renames only the scalar accumulators into free/other registers) -> 82 conflicts.
Output bit-identical (0 of 17.8M differ). Harness 8704x5120 N=2048: 14.40 -> 14.03 ms (-2.6%, incl.
dequant+prescale). So bank conflicts cost ~3%, not the whole gap. Under load the card sits at
~1265-1278 MHz, 175 W (power-throttled): peak there 18.2 TFLOPS; kernel alone ~13.5 TFLOPS = ~74%.

## 2026-09-28: fold GEMM ablation (harness, 8704x5120 N=2048, ms incl. ~0.45 ms dequant+prescale)

| variant (timing only) | ms |
|---|---|
| shipped | 14.44 |
| no fp32 folds | 14.07 |
| no tile store + barrier | 12.43 |
| no global loads (stores keep stale data) | 13.67 |
| no global loads, no barrier | 12.57 |
| no barrier only | 14.17 |
| no store/barrier/loads/folds (pure LDS+HFMA2) | 12.09 (~86% of the throttled peak) |
| bank-renamed shipped kernel (bit-identical) | 14.03-14.14 |

Tried (exact, both slower): spreading the 16 STS into k2 = 8..15 (15.9); fetching two tiles ahead,
STS then LDG(it+2) before the barrier (15.7; 15.3 even after bank renaming). The ~14% store/barrier
loss is real but the obvious CUDA-level reorderings lose more than they save.
Attention kernels: fa_fold_qk2 45% of HFMA2 bank-conflicted (REG 128, 2 CTAs/SM: renaming must stay
inside the accumulators), fa_fold_pv 36% (REG 214). Renaming worth ~3% on the GEMM.

## 2026-09-28: nsys trace of pp2048 d0 (two passes, per pass per GPU)

kernels 4.56 / 4.61 s; peer copies (kind 10) 125 x 21 MB at 7.0 (GPU0) / 8.3 (GPU1) GB/s = 0.37 / 0.32 s,
zero overlap with kernels; idle ~0.1 s. Exposed exchange = ~7% of a prefill pass at any depth.
Exact fix: token-chunk the row-parallel matmuls (attn out, FFN down) and start each chunk's peer copy
while the next chunk computes (2 chunks ~3.5%, 4 chunks up to ~7% minus GEMM wave-tail losses).

Exact 0-context budget (392 t/s now): bank renaming ~2%, exchange overlap 3.5-7%, chunked delta-net
~5%, GEMM store/barrier phase up to ~12% (not yet solved). All four: ~490; realistic ~440-460.

## 2026-09-28: exchange overlap landed (OPTLOG 237)

pp2048 d0 392 -> ~411 t/s (+4.5%), logits bit-identical (3.05 GB compare). Direct P2P epilogue writes:
253 t/s (dead end, opt-in only). Chunked delta-net graph (build_delta_net_chunking) asserts under
-sm tensor (meta split axis unknown); a chunked GDN needs its own CUDA kernel (<= 7.5% available).
Recurrent GDN kernel already swept (OPTLOG 73, 76): latency-bound on per-token warp reductions.

Remaining exact budget at 0 context from ~411: GEMM store/barrier phase (<= ~10% overall, unsolved),
bank renaming (~2%, harness-only so far), chunked GDN kernel (<= ~5-7%). Best case ~490, i.e. 500 not
yet in reach with what is known; 300 at 260k remains above the P100's peak FLOP rate for exact math.
SASS hand edit (movests.py): the 16 STS + 3 DEPBARs moved from the store phase into the HFMA2 block
(55/70/85% points), addresses recomputed at the loop top into R200-R204. Bit-identical, but 14.73 ms
against 14.53 for the unmodified cubin: store placement is not the cost. The ~14% is the global-load
latency interacting with the per-tile barrier; fixing it needs a restructured pipeline (e.g. a
3-stage smem ring or warp-specialised loads), not a reordering.
Warp-specialised GEMM (fold_kernel_ws.cuh: 8 compute + 2 loader warps, 3-stage smem ring, named
barriers bar.arrive/bar.sync): bit-identical, 14.87 ms vs 14.45-14.57 (-2.5%). With the store move
also not helping, the ~14% of the no-barrier ablation looks like the benefit of warps drifting out of
lockstep (staggered LDS bursts), which any per-tile handoff re-synchronises. Open idea: stagger the
warps deliberately (e.g. half the warps start at k2 = 8) with a 3-stage ring so no CTA-wide barrier
is needed per tile.

## 2026-09-28 night: team results (agents hit the rate limit ~25 min in; salvaged by hand)

- GEMM (team/gemm/k3.cuh) -> gemm_fold_kernel_u2, committed (OPTLOG 238): pp2048 d0 ~413 -> ~449 t/s
  (+8.5%), gates pass (tg256 32.17, PPL 2.6101). Not bit-identical only because the old fold lost values
  to FTZ under -use_fast_math; NMSE vs fp64 equal.
- Delta-net (team/gdn/gdn_chunked_v2.cuh + harness_v2.cu, precision knobs TG/TA/TB/TC/TF): all-float
  1.75-1.92 ms vs shipped 6.6-7.3 (3.8-4.2x). Accuracy vs fp64: mode 0 4x better; mode 3 (model-like)
  ~tied (out nmse 3.40e-14 vs 3.35e-14, state maxabs/rms 2.8e-6 vs 2.0e-6). All-double: 4.37 ms, still
  not better on every metric. NEXT: check modes 1-2 / more seeds, then integrate into gated_delta_net.cu
  (interface gdn_ch_params / gdn_ch_launch, scratch per chunk) and judge on real PPL/KLD.
- Attention (worktree team/wt-attn, uncommitted diff in fattn-gemm.cu): pv2 (128-thread PV, 2 CTAs/SM)
  + 2-stream head split: FLASH_ATTN_EXT kv 65536 nb 1024 68.3 -> 66.6 ms (~2.5%). Op runs ~12 TFLOPS.
- Open question: 260k measured 106 t/s but kernel rates predict ~150; the gap is outside the FA kernel
  (suspects: MTP draft prefill at -ubd 64 re-reading the full KV per 64 tokens, sustained clocks). Profile
  a depth run before more kernel work.

## 2026-09-29: session summary (pp2048 -ub 2048: 437 -> ~482 t/s; 260k: ~123 -> ~130)

Kept, all gated (exact = KLD at the base floor -0.000006 / 0.000004 / 100% unless noted):
- 239 chunked delta net (KLD vs fp64 tied with the recurrence): +6.3%
- 244-247 gate/up pairing, exchange-wait weight prefetch, high-priority copy stream, one-pass prescale: ~+2.8%
- 248 compact causal KQ mask: -ub 2048 now fits at 262k with vision (GPU0 min free 732 MiB vs 508 at -ub 1024 before)
- 249 fold-attention skip of fully masked tiles (exact); fold path at 0 context REJECTED (8-49x worse NMSE vs fp64)
- 250 warp-per-row RMS norm (+0.9%), 251 tiled transposed concat (+0.3%)
Rejected: k8/k9 GEMM variants, exchange splits, 2-stream chunks, side-stream prefetch.
User rules added: -ub 2048 is the max; it must fit vision + 262k with headroom (now true).

Next (in progress): register-bank-fixed SASS for gemm_fold_kernel_u2 (GEMM is 83% of the pass).
Harness: +1.6-2.4% GEMM, bit-identical by construction (bankfix.py renames registers only).
- generator: p100-handoff/tools/sass-gemm/u2cubin.py (writes ggml/src/ggml-cuda/gemm-fold-u2-sass.h; not run yet)
- runtime loader + 3 launch sites: p100-handoff/wip/u2-sass-launch.patch (apply after generating the header)
- then: KLD floor check, pp2048 A/B, tg check, commit. Also queued: SwiGLU fused into the down prescale (~0.5%).
260k: attention is 64% of the time at ~11 TFLOPS; exact ceiling ~150-160, 200 is not reachable with exact math.

## 2026-09-29 (later): bank-fixed u2 SASS landed (pp2048 -ub 2048 ~485 -> ~491 warm, ~500 cold)
- 252 KEPT: u2cubin.py + runtime cuModuleLoadData loader (gemm-fold-u2-sass.h). Exact (KLD floor), +1.2%.
  gate.sh fails if the header is stale: after ANY edit to the u2 kernel text, rerun
  `python3 p100-handoff/tools/sass-gemm/u2cubin.py` (~40 s) then rebuild ggml-cuda (~20 s).
  GGML_CUDA_GEMM_FOLD_SASS=0 falls back to the compiled kernel.
- 253 REVERTED: a/b slot swap + reuse-flag rewrite; the adjacent-only reuse model is wrong (ptxas relies on
  the operand cache lasting 2-3 instructions). Left: 170/399 conflicts are A x B fragment pairs (LDS.128
  groups, not renameable); only instruction reordering could remove them.
- Temperature matters: single pp2048 passes read 502 at 41C and 487 at 60C on the same build. A/B only
  interleaved (tools/ab-pp.sh A B B A; LD_LIBRARY_PATH=<dir with other libggml-cuda.so> as one arm).
- NEXT: SwiGLU fused into the down-projection prescale (~0.4%, exact). Design issue found: the GLU node must
  only be skipped when the next MUL_MAT is sure to take gemm_fold_try (dispatcher picks cuBLAS/fold for this
  shape, but that must be decided at the GLU node, or materialize the GLU in every non-fold path).
- 254 KEPT: SwiGLU fused into the fold prescale (exact, +0.5%): pp2048 -ub 2048 ~500.5 in an interleaved A/B
  at 44-54C (off 498.0). 260k serving run: prompt 128.4 t/s, GPU0 min free 732 MiB (fit unchanged), no assert.
  Full suite (gate.sh --full) not yet run on 252-254. 260k: exact ceiling ~150-160; 200 needs non-exact math.
- NEXT LEAD (op profile, GGML_CUDA_OP_PROFILE=2, pp2048 -ub 2048, warmup+run): fused:RMS_NORM n=24 [norm-N]
  (delta-net gated output norm) = 452 ms / 96 calls = 4.7 ms each, 5.9% of op time, for ~25 MB of data.
  Either it absorbs the chunked GDN's async work (GDN itself shows only 1.9 ms/call) or the fused norm
  kernel is badly shaped for [128, 24, 2048]. Isolate with test-backend-ops perf / nsys first; if real, ~5%.
  Top ops: fold GEMMs 5120x8704 40.8%, 8704x5120 21.7%, 5120x5120 9.6%, 3072x5120 7.8%.
  CONFIRMED 21:26: with GGML_CUDA_GDN_CHUNKED=0 the norm still reads 449.9 ms / 96 -> the fused norm chain itself
  is slow (~4.7 ms for [128, 24, 2048]; memory time should be ~0.1-0.2 ms). Real target: ~5% of the pass.
  Suspects: fused rms_norm+mul(+mul) with a strided/broadcast operand (gate z view) read uncoalesced, or a
  launch geometry that bypasses the warp-per-row path. Start: dump the fused node's srcs (ne/nb) and
  reproduce in test-backend-ops perf -o RMS_NORM.
