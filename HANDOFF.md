# Handoff

Where the work stands, and what's worth doing next. For the project rules and gates, see
`CLAUDE.md`. For the results and changes, see `p100-docs/`.

## State (2026-09-23)

- Branch `p100-optimizations`, merged with upstream `f46bc30cb`. See CHANGES §9 for what the
  merge touched and how it was checked, and §10 for the real-world MTP work since.
- `tg256` 31.3 t/s (upstream at the fork point: 17.51). Perplexity 2.6101 ± 0.0198 on the gate corpus.
- Real-world MTP, ms per cycle through `llama-server` (`tools/depth-bench.py --restore`): ~73 at
  2k, ~87 at 64k, ~110 at 128k, ~143 at 260k. That is 38-50 t/s at 2k and 24-27 t/s at 260k,
  depending on how much of the text the draft predicts. The slot snapshots it restores are in
  `/mnt/fast/p100-scratch/slots`, so a full-context measurement takes minutes, not an hour of prefill.
- The release bundle at `/mnt/fast/p100-llamacpp-release` is refreshed from this branch.
  `diffs/HEAD-SHA.txt` there is authoritative.
- ~190 attempts are logged in `OPTLOG.md`. Its CLOSING SUMMARY (line ~1896) is from 2026-09-01 and
  predates the long-context work. The later attempts are the current story.

## Where the MTP cycle goes (nsys, 2k context, n_max 4)

~64 ms of GPU work in a ~72 ms cycle. The 5-token verify is 56.8 ms of GPU time, nearly all of it
the 5-column q6_K matvec, which is at its floor for exact arithmetic (OPTLOG 179, 187). The draft
steps and catch-up are 7.4 ms. The remaining ~8 ms is host time:

- a ~200 µs skew at the start of every graph, because one host thread enqueues GPU0's subgraph
  before GPU1's and GPU0 then waits for GPU1 at the first all-reduce;
- the draft context rebuilding its graph twice per cycle (catch-up is 5 tokens, a draft step 1),
  now under 1 ms each;
- ~9 input uploads per decode, each syncing both GPUs, because the meta backend has no events.

nsys works on Pascal (2022.4). If the importer fails, run
`/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i X.qdstrm` and then
`nsys export -t sqlite`. `perf` is locked here (`perf_event_paranoid` 4). For a host CPU profile
use `tools/pmp/`, an LD_PRELOAD sampler; its header has the usage.

## Open threads

1. **Per-GPU enqueue threads in `ggml-backend-meta.cpp`.** This would fix the skew above, and it
   would double the enqueue rate, which matters in the draft steps. It needs host-side ordering
   between the two threads at every all-reduce (the peer copy's event must be recorded before
   the other side's stream waits on it). Worth maybe 2-4% at short context.
2. **One cached graph per batch shape.** The scheduler and the meta backend each keep only the
   last graph, so any shape change rebuilds it: ~20 ms for the target (a8b274ea6 got it there
   from ~45), under 1 ms for the draft context, which rebuilds twice per cycle. With a graph
   per verify width, a confidence-based draft length (OPTLOG 192, `LLAMA_SPEC_LOG` has the
   data) simulates at −11% ms/token at 2k, −15% at 64k and −22% at 260k. That needs one
   scheduler (small compute buffers) per width, plus a per-uid subgraph cache in the meta
   backend. Its external-view containers rotate two-deep today, which is the tricky part.
3. **A possible timing-dependent result in an earlier binary.** One build gave three different
   260k texts across normal and profiled runs. The current build agrees with itself, async and
   under `CUDA_LAUNCH_BLOCKING=1`, on every case tried. See OPTLOG 192. A repeat-until-diverge
   test at 260k would settle it.
4. **q4p occupancy.** Every variant runs one 256-thread block per SM (233-255 registers). At 30
   rows the PV accumulators alone are 120 registers. A design that keeps fewer rows per thread,
   or splits PV across two blocks, might approach the ~70% FFMA efficiency the instruction mix
   allows, against ~44% now. At 30 rows it is latency-bound (same time at 1189 and 1328 MHz),
   so more loads in flight should matter more than fewer instructions. Measure in the server,
   not only in test-backend-ops (OPTLOG 190).
5. **Short-prompt prefill (time to first token per chat turn).** A 9-127 token prompt takes
   ~0.5 s of GPU at any depth: cuBLAS's 256x128-tile HGEMM at ~4 TFLOPS on a skinny GEMM, plus a
   full f16 dequant of the weights each call (OPTLOG 195). A narrow-tile kernel would help. Any
   replacement must match ALGO6's accuracy (attempt 153).
6. **`GGML_CUDA_DEVICES` above the physical GPU count isn't reproducible** (NaN in 4 of 8 runs at
   3 virtual devices). It follows the GEMM attention path. It's debug-only, and two physical GPUs
   are bit-stable. OPTLOG attempt 153 §8c.
7. **Fuse the all-reduce widen into the ADD** (~+1% prefill). It needs an accumulating-copy path
   in `ggml-backend-meta.cpp`.
8. **`gated_delta_net`** is 7% of prefill and at ~15% issue efficiency. It resisted three attempts.
9. **Deepest prefill regressed ~10%** (95.1 → 85.4 t/s at `-d 262144`). Possibly thermal; not
   bisected.

## Closed: don't re-sweep without new information

| axis | result |
|---|---|
| attention occupancy (384 threads, occupancy 2-4, doubled warps) | neutral or worse, three ways |
| `nbatch_K` 64/128/256 | 128 is best on narrow tiles; 256 is +44% |
| `nbatch_fa` 32/64/128 | 64 |
| Q-column reuse (`cpw` 1 vs 2) | identical to 0.005% |
| wide loads in the q4_0 dequant | scalar wins by 7% |
| mmvq register caps, unrolling, prefetch pipelines | all worse; not latency-bound |
| double-buffered mmvq staging | shared memory then limits occupancy |
| internal AllReduce on Pascal | −17% (PCIe); re-measured exact (no BF16 wire) in OPTLOG 189: still ~1% slower |
| CUDA graphs for small batches only | no gain under `-sm tensor` (OPTLOG 186) |
| 5-column q6_K matvec: fp32 or fp16 rewrites, I2F removal | at its floor (OPTLOG 179, 187) |
| MMQ on Pascal | no DP4A, ~4x ALU disadvantage |
| `-sm layer`, for prefill or decode | tensor split wins both |
| f32 GEMM output | halves GEMM throughput |

## Notes for kernel work

- The mmvq geometry and staging live in `mmvq.cu`, under `GGML_CUDA_MMVQ_PASCAL`. Build times:
  `mmvq.cu` alone is ~90 s, while touching `vecdotq.cuh` rebuilds ~200 instances (~25 min).
  Put sweep knobs in the source, not in `-D` flags.
- Fast isolated benchmark (a few seconds):
  `test-backend-ops perf -o MUL_MAT -b CUDA0 -p 'type_a=q6_K,type_b=f32,m=4096,n=1,k=14336'`.
  Always confirm on the real model. The isolated shape has pointed the wrong way before.
- Casting a shared-memory pointer through `uintptr_t` loses the address space, and ptxas silently
  emits generic loads. Derive aligned pointers with `char *` arithmetic.
- Correctness harnesses and proofs are in `p100-handoff/tools/` (bit-exactness replays for the
  fastdiv, DP4A, q4_0 dequant and norm changes). `p100-handoff/VERIFICATION.md` is the numerical
  audit.
