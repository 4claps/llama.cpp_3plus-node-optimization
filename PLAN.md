# PLAN: Kmic-68 P100 work on 3+ GPUs

Research and planning only. Nothing here has been built or run. Prepared 2026-10-02 against this
checkout at `ae35056eb` (identical to Kmic-68 `p100-optimizations`, tag `p100-optimizations-b11515-ae35056`).

## Executive summary

1. *(Corrected 2026-10-03, see section 1.3: PXA runs a P2P tensor split on 4× P100 over PCIe; 3 cards are still unmeasured anywhere.)* Nobody has published an N-way, Pascal, PCIe-only P2P AllReduce validated on 3 real GPUs. The closest prior art is ik_llama.cpp's `reduce.cu` (N-way one-shot plus ring, and already the source of this fork's 2-GPU kernel). Port from it; don't adopt another fork wholesale.
2. The `internal AllReduce init failed (n_devices != 2?)` warning is not evidence of a 3-GPU problem. It prints on 2-GPU P100 boxes too, because upstream's internal AllReduce rejects compute capability below Volta. The real 3-GPU signal is that the fork's own "one-kernel P2P AllReduce" INFO line never prints on the test machine: that path requires exactly 2 backends.
3. On 3 GPUs, every exchange (about 128 per token) falls to the meta-backend butterfly. For N=3 that is three dependent stages: fold 2→0, swap 0↔1, then copy 0→2. The fork's 2-GPU overlapped prefill exchange is lost too.
4. The AllReduce theory is plausible but unproven. On 2 GPUs the fork measured its P2P kernel at only +1.3% over the butterfly, and at least four other 3-GPU or machine-specific factors exist.
5. Those factors: the fork's published numbers come from a pure Q6_K (every quantized tensor Q6_K, 20.88 GiB, apparently self-quantized; no published file matches), while the test machine's UD-Q6_K_XL streams only 38.9% Q6_K per token (56.6% Q8_0) and 13.6% more bytes; the fork's fp16 verify and fused gate/up kernels are Q6_K-only (the fused path covers 21 of 64 layers); on 3 GPUs the 4 KV head-groups split 2/1/1; and the host is a 2011 Sandy Bridge-E.
6. Test machine topology: P2P read and write report OK on every pair (all PHB), IOMMU is off, and all three cards run Gen3 x8 (not two). Sandy Bridge Xeon platforms are documented at about 800 MB/s for P2P reads, so designs should push (write) rather than pull. This must be measured.
7. The decisive first experiment (Phase 1, about one day of GPU time, no code changes): the same build on the test machine with 2 GPUs versus 3, and a pure Q6_K (the fork's reference quant, reproduced locally) versus UD-Q6_K_XL, plus nvprof attribution of exchange time and a P2P read/write microbenchmark.
8. Gate: build a custom N-way AllReduce only if exchange plus cross-GPU wait is at least 20% of decode or 64k-prefill time and P2P writes perform on all pairs. Below 10%, stop and pivot to quant and kernel work.
9. Cheap levers to test first: an NCCL build (NCCL is already in the image; exact for decode, but BF16-lossy for prefill as written), skipping the unused token-chunked GEMMs when N≠2, `-sm layer` versus `-sm tensor`, and the choice of quant.
10. Ranked designs: (1) measure and pull the cheap levers; (2) N-way one-shot push AllReduce for decode plus a chunked reduce-scatter/all-gather for prefill; (3) Q8_0 kernel parity; (4) KV-group rebalancing; (5) per-device enqueue threads; (6) grouped TP2; (7) TP2 plus a spare card. Each phase ends at an approval point.

Conventions: **[measured]** = observed on the test machine or in a cited source; **[code]** = read in this
checkout; **[inferred]** = my reasoning, not yet measured.

---

## 0. Correcting the starting premise

The task background treats the log line `internal AllReduce init failed (n_devices != 2?); falling
back to meta-backend butterfly` as proof that the fork's custom P2P AllReduce is off on 3 GPUs. The
conclusion is right, but the evidence is not:

- **[code]** That message comes from upstream's internal AllReduce (`ggml/src/ggml-cuda/allreduce.cu`,
  upstream PR #22299). Its init rejects `n_devices != 2` (`allreduce.cu:402`) **and** any device with
  `cc < GGML_CUDA_CC_VOLTA` (`allreduce.cu:412`), because its spin loop uses `__nanosleep` (sm_70+).
  The P100 is sm_60, so this path fails on **any** P100 setup, including 2 GPUs.
- **[code]** The fork's own docs agree. OPTLOG attempt 128: "The real reason for the fallback is not
  n_devices — it is `ggml_cuda_ar_pipeline_init` rejecting `cc < GGML_CUDA_CC_VOLTA`." The fork's
  2-GPU box prints the same warning on every run.
- **[code]** The fork's custom path is separate. "Butterfly" is a wrapper
  (`ggml_backend_cuda_comm_try_allreduce_butterfly`) that calls the fork's
  `ggml_backend_cuda_comm_try_allreduce_p2p`, which returns false when `backends.size() != 2`
  (`ggml-cuda.cu:1315`). Only then does the meta backend run its generic fallback.
- **[measured]** The real tell on the test machine: when the fork's P2P path is active it logs `small
  tensor-parallel exchanges use the one-kernel P2P AllReduce` (`ggml-cuda.cu:1342`). That line is
  absent from the round-4 battery server log on the test machine.

So on 3 GPUs, every tensor-parallel exchange runs the meta backend's generic fallback (section 2.2).

## 1. Prior art

### 1.1 Findings

| # | Source | Date | What it does | 3+ GPUs | Pascal | PCIe-only |
|---|---|---|---|---|---|---|
| 1 | [ggml-org/llama.cpp PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378): backend-agnostic tensor parallelism (the meta backend, `-sm tensor`) | opened 2026-02-05, merged 2026-04-09 | Splits the graph into subgraphs. After each one, partials are exchanged with peer copies plus one-node ADD graphs (a butterfly; non-power-of-2 N folds the excess first and copies back last). The PR says N is arbitrary (capped at 16), NCCL is recommended, and the generic path is "presumably still suboptimal vs. NCCL". | Yes, generically | Yes (generic) | Yes |
| 2 | ggml-org/llama.cpp PR #22299 "internal AllReduce kernel for CUDA provider" (commit `f3c3e0e9a`) | 2026-05-10 | Host-staged, one kernel per GPU, pinned-memory flags. Copy-engine path for large tensors. | **No** (n=2 only) | **No** (Volta+) | Yes |
| 3 | Kmic-68 commit `defe84bd7` (OPTLOG 230), plus the chunked prefill exchange (`ggml_cuda_allreduce_chunked`, OPTLOG 237: +4.5% pp2048, bit-identical) | 2026-09-26 | One-kernel direct-P2P AllReduce for f32 tensors ≤256 KB: GPU j sums half j and writes both. Prefill exchanges are token-chunked GEMM output sent f16 on a copy stream, overlapped with the next chunk. Measured +1.27% tg128 (30.58→30.97 t/s) and +0.96% verify on 2×P100 (PHB). | **No** | Yes | Yes |
| 4 | [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) `ggml/src/ggml-cuda/reduce.cu`; [PR #1022](https://github.com/ikawrakow/ik_llama.cpp/pull/1022) (graph split POC), [PR #1080](https://github.com/ikawrakow/ik_llama.cpp/pull/1080) "Graph parallel: the next generation" | Nov 2025 – merged 2025-12-24 | **N-way.** Decode-sized tensors (`ne[1] < 32`, P2P on): a one-shot where each GPU owns 1/N, **reads** all N partials directly over P2P, sums, and writes the sum to every GPU (`k_reduce_add_T<..., nptr>`, specialized for 2/3/4). Large tensors: ring reduce-scatter + all-gather over `cudaMemcpyPeerAsync` (2(N−1) stages; source comments walk through N=2, 3, 4). Optional NCCL, with pairwise communicators for 3-4 GPUs because one communicator performed poorly. PR #1080 reports results on 4×RTX 3090 without NVLink. | **Yes** | Kernels have no arch gate (read in source); ik_llama's overall Pascal support unverified | Yes |
| 5 | [Joe11221/p100-llama-cpp](https://github.com/Joe11221/p100-llama-cpp) patches 0003, 0004, 0006 | pinned to llama.cpp b10660 | 0003: upstream internal AR on sm_60 (a `clock64()` spin instead of `__nanosleep`). 0004: separate H2D stream. **0006: extends the host-staged internal AR to N devices.** Prefill gains on a **dual-socket** Xeon Gold 6148 host. MIT. Not upstreamed, by the author's choice. | Only **virtual** devices (2×P100 split into 3 and 4); perplexity matched | Yes | Yes |
| 6 | Kmic-68 OPTLOG 128 and 189 (same idea as #5, tried on the fork's box) | — | Host-staged internal AR on Pascal: attempt 128 rejected at **−17%**; attempt 189 exact but "not faster" (tg64 30.98 vs 31.32 butterfly). "The staging through host memory costs more GPU latency than the API calls it saves." OPTLOG 235: Joe11221's gains "are from a dual-socket host". | n/a | Yes | Yes |
| 7 | [vLLM custom all-reduce](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/device_communicators/custom_all_reduce.py) (origin: [PR #2192](https://github.com/vllm-project/vllm/pull/2192)) | current main | One-shot and two-shot over P2P IPC buffers. **Explicitly disabled for "more than two PCIe-only GPUs"**; `_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16]` (3 excluded); falls back to NCCL. | **No** (on PCIe) | Not targeted | 2 GPUs only |
| 8 | [Mikec78660/vLLM-Pascal](https://github.com/Mikec78660/vLLM-Pascal) | — | vLLM for P100/P40, Qwen3.8 27B validated on 2×P100. NCCL AllReduce at TP=2 is a per-step cost. | No | Yes | Yes |
| 9 | [local-inference-lab/rtx6kpro: pcie-oneshot-allreduce.md](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md) | — | **Push** design: every GPU writes its data to every peer over P2P, then a system-scope barrier, then a local reduce. Double-buffered, fused with RMSNorm. 4 GPUs: 7.7 µs vs NCCL 14.7 µs at 16 KB, 11.8 vs 71.2 µs at 64 KB, 28.7 vs 40.9 µs at 256 KB; NCCL wins above 512 KB. Notes the driver may stage P2P through system memory unless P2P settings are forced. | Tested at 4 and 8 (not 3) | No (SM120) | Yes |
| 10 | [local-inference-lab/b12x issue #410](https://github.com/local-inference-lab/b12x/issues/410) "PCIe oneshot all-reduce for world size 3" | opened 2026-09-21, open | The world-size allowlist is even-only (2, 4, 6, 8, 10). The author says the algorithm handles odd N in principle and that NCCL fallback took about 40% of decode kernel time. | Wanted, not done | No (SM120) | Yes |
| 11 | TensorRT-LLM custom AR (one-shot, two-shot, Lamport); [vLLM PR #2192](https://github.com/vllm-project/vllm/pull/2192) | — | NVLink/Hopper+ focus. Kmic-68 OPTLOG 235 lists "Lamport push AllReduce with a fused ADD+RMS_NORM epilogue (TRT-LLM)" as its next step, not done. | NVLink | No | No |
| 12 | Papers: ["Every Microsecond Matters" arXiv 2607.16100](https://arxiv.org/abs/2607.16100) (2026-07-17); ["SiFAR" arXiv 2607.08973](https://arxiv.org/abs/2607.08973) (2026-07-09) | 2026 | Barrier-free sync and symmetric memory near speed-of-light (scale-up fabrics). SiFAR: synchronization-free AR, dual buffering, evaluated on 8×H200, uses in-switch reduction. Ideas transfer (barrier-free, double buffering); hardware does not. | Yes | No | No |
| 13 | [krampenschiesser/llama.cpp PR #3](https://github.com/krampenschiesser/llama.cpp/pull/3) `--max-tensor-split N` | 2026-09-29 | Groups GPUs into tensor-parallel groups of at most N, and distributes layers across groups. No performance data. | Yes (grouping) | Not stated | Not stated |
| 14 | [NVIDIA: Benchmarking GPUDirect RDMA on modern server platforms](https://developer.nvidia.com/blog/benchmarking-gpudirect-rdma-on-modern-server-platforms/) | not retrieved (page fetch timed out; quote from search index) | "Testing the previous generation platform (Sandy Bridge Xeon), the peer-to-peer reading bandwidth seems even more limited, in this case around 800MB/s." Measured for a NIC reading GPU memory; applying it to GPU↔GPU reads on the test machine is **[inferred]**. | — | — | Platform caveat |
| 15 | [Hackaday: "Getting MTP and Multi-GPU Tensor Splitting Working on Tesla P100s"](https://hackaday.io/project/205768/log/250966) | 2026 | 3×P100, Qwen3.8 27B, MTP + tensor split. A search snippet quotes ~17.3 t/s tensor vs 8.8-10.9 layer. **The page returned HTTP 502 twice, so this is unverified.** It may be your own homelab log. | Yes | Yes | Yes |
| 16 | [ggml-org/llama.cpp issue #29466](https://github.com/ggml-org/llama.cpp/issues/29466) | opened 2026-09-26 (by 4claps), open | `-sm tensor` second-request crash `GGML_ASSERT(bcj.nodes[i])` on 2× and 3× P100. Not a performance item, but every change below must not make it worse. | — | — | — |
| 17 | [poisonxa16/pxa](https://github.com/poisonxa16/pxa) (MIT engine, forked from ik_llama.cpp at `1520eda98056`; read at `232fd83`, 2026-10-03). See section 1.3 | v2026.10, 2026-09-27/28 | Pascal/Volta engine with its own quant formats. Tensor split on 2 matched cards and on **4× P100** by default, with a one-kernel P2P all-reduce that runs on sm_60, a 2-card chunked prefill push, and an NCCL fallback. **3 cards are explicitly unmeasured and stay on `-sm layer`.** | **4 yes; 3 no** | **Yes** | Yes (x4 links) |

Also checked: Kmic-68 has issues and discussions **disabled** and no roadmap beyond `HANDOFF.md`. Its only
PR is [#1](https://github.com/Kmic-68/llama.cpp/pull/1) (Q4_K mmvq on Pascal, by mewsian). Its four forks
(laurentiuluca, mewsian, BlackPavilion, and this one) contain nothing beyond Kmic-68 tags. Upstream has no
open or merged PR generalizing the CUDA AllReduce to N>2 as of upstream master `bed0a8566`. The related open PRs are
multi-node NCCL ([#28967](https://github.com/ggml-org/llama.cpp/pull/28967)) and a Vulkan CPU-proxy
AllReduce ([#25051](https://github.com/ggml-org/llama.cpp/pull/25051)). The fork's `HANDOFF.md` open
threads list per-GPU enqueue threads (#1), device-chained MTP draft steps (#2), and NaNs in 4 of 8 runs with
**3 virtual devices** on the GEMM-attention path (#6). Nothing about 3 physical GPUs.

### 1.2 What is done, partly done, and not done

- **Done:** Tensor split across any N in llama.cpp (the meta-backend butterfly, generic and slow). N-way
  ring plus one-shot reductions in ik_llama.cpp (PCIe 3090s, a different codebase).
- **Partly done:** An N-way **one-shot direct-P2P** AllReduce exists (ik_llama.cpp) but it is
  pull-based (remote reads), sits in a codebase without this fork's Pascal kernels, MTP and verify
  work, and is not validated on Pascal or a Sandy Bridge-E root complex. An N-way **host-staged**
  AllReduce for Pascal exists (Joe11221 0006), but only on virtual devices, and Kmic-68 measured
  host staging as no faster than the butterfly on its PCIe box.
- **Not done (that I found):** a push-based (write-only), latency-optimized N-way AllReduce validated on
  Pascal, PCIe-only, at odd N=3; the fork's overlapped (token-chunked) prefill exchange generalized to N>2;
  any 3-GPU measurement of this fork beyond our own round-4 run.
- **Confidence:** moderate. The main projects were covered directly (code read for upstream, this fork,
  ik_llama; vLLM source; GitHub API for Kmic-68 forks and upstream PRs). GitHub search is not exhaustive,
  ik_llama's Pascal status is unverified, and the Hackaday log couldn't be fetched.
- **Correction 2026-10-03 (section 1.3):** PXA runs a P2P tensor split on 4 real P100s over PCIe and ships a
  one-kernel all-reduce for sm_60; 3 cards remain unmeasured there. Its `reduce.cu` is now the closer porting source.
- **Recommendation:** nobody has solved this in a form you can adopt as-is. Don't switch to ik_llama.cpp:
  you would lose every Kmic-68 Pascal kernel, the MTP/verify work and the correctness gate. **Port ik_llama's
  N-way reduction structure (MIT) into this fork's comm path**, adapted to push rather than pull if the test machine's P2P
  reads are slow, and only if Phase 1 shows the exchanges are worth it.

### 1.3 PXA (added 2026-10-03; research only, nothing run)

Source: [poisonxa16/pxa](https://github.com/poisonxa16/pxa/blob/232fd83) at `232fd83`. Read: `README.md`, `docs/DELTA-SINCE-IK.md`, `docs/KNOWN-ISSUES.md`,
`docs/DEFAULTS.md`, `docs/LEVERS.md` (tensor-split rows), `docs/LAUNCHER.md` (searched, not read end to end),
`bench/fair-battle.md`, `bench/KLD-RESULTS.md`, `RELEASE-NOTES-2026-09-07.md`, `RELEASE-NOTES-v2026.10*.md`,
`docs/PXQU-FLASHNEXT.md`, `docs/FLASHNEXT-RESEARCH.md`, `docs/PXQU-CONVERT.md`, `docs/QUANTIZING.md`,
`docs/PASCAL-DECODE-GAP.md`, `docs/PXA-SM60-SERVING.md`, `ggml/src/ggml-cuda/reduce.cu`, `src/pxa-tsplit.h`,
`common/pxa-registry.cpp`, `examples/server/server-context.cpp`. **Not found:** `docs/ENGINE.md` does not exist
in the repo, and the two Flash-Next files live under `docs/`, not the root. GitHub has 2 open issues
([#3](https://github.com/poisonxa16/pxa/issues/3) missing `.env`, [#4](https://github.com/poisonxa16/pxa/issues/4) GPU detection
in a container) and 2 closed PRs (#1, #2), none about multi-GPU; the discussions page returned a load error and
could not be enumerated. The "bug #206"-style numbers in its docs are an internal tracker, not GitHub issues.

**1. 3 GPUs, tensor split, N-way all-reduce, P2P on Pascal.**
- PXA's tensor split is ik_llama.cpp's graph split (`GGML_OP_REDUCE`), not upstream's meta backend. Its routes are in
  [`reduce.cu`](https://github.com/poisonxa16/pxa/blob/232fd83/ggml/src/ggml-cuda/reduce.cu):
  - **Fused one-kernel all-reduce** (`PXA_TSPLIT_REDUCE=fused`, lines 413-481; default for decode on a pair). Each
    device copies its partial to its own device-memory staging ring, publishes an arrival token, spins on the peers'
    tokens through the P2P mapping, then sums the peers' staging slots in place. **Pull design, one launch per device,
    no events.** It backs off with `clock64()` below Volta, so it runs on sm_60 (lines 473-476).
  - **N>2 is refused on the fused route unless `PXA_TSPLIT_REDUCE_NWAY=1`** (line 964), because its summation order
    differs from the older p2p-direct route at 3+ devices (lines 467-471).
  - **Epilogue fusion** (`PXA_TSPLIT_EPI`, `_EPI_PUSH`, `_EPI_NORM`; `docs/LEVERS.md:62-64`): the residual add, the
    cross-card sum and the next RMS norm run in the reduce kernel, and the weight kernel writes its partial straight
    into the peer's staging slot.
  - **2-card prefill** (`PXA_TSPLIT_PF`, lines 1984-2011): one copy-engine push per direction in K chunks on a copy
    stream, adds overlapped. Partials are f16 at ≥32 tokens.
  - **N>2 prefill** takes ik's ring (reduce-scatter + all-gather, lines ~2305-2330).
  - **NCCL** serves all-device reduces at 2 cards or decode width. On PXA's box NCCL picks **SHM because every pair is
    PHB** (lines 1921-1924), as on the test machine. A 4-rank SHM group does not fit Docker's 64 MB `/dev/shm`; the group failed
    silently and decoded garbage (their bug #206); it now falls back to the in-tree peer route.
  - A startup **P2P self-test** per pair (`docs/KNOWN-ISSUES.md:5-11`), `PXA_P2P=0` to stage through host.
- **3 cards: nothing.** "4x V100, 3 cards, 5+ cards. The tensor split has no measurement on these sets, so they keep
  layer" (`docs/DEFAULTS.md:92`; `common/pxa-registry.cpp:590-591`; `RELEASE-NOTES-v2026.10.md:181-182`).
- **4× P100 seat: tensor split by default since 2026-09-27** for the dense 27B (`src/pxa-tsplit.h:26-35`): tensor vs
  layer on four cards measured decode +4.6–5.6% (PXQN4) / +38.5–41.1% (PXQ4) and prefill at 22.6k +95%. README:
  4× P100 PXQN4 prefill 440, decode 30.7 t/s, against 2× P100 345 / 37.8; "our 4x P100 test rig runs every card on a
  x4 PCIe link, so four cards trade decode for prefill". Before v2026.10 the four-card seat (the 177B MoE) ran
  **layer split**, and the launcher default for that file stayed the layer recipe (`RELEASE-NOTES-v2026.10.md`,
  Flash-Next paragraph). So: dense 27B = tensor; the large MoE seat = layer by default, tensor available.

**2. The ten documented negatives** (`RELEASE-NOTES-2026-09-07.md:727-744`), with the number that killed each:
  1. Volta register-direct PXQ4 GEMM: 40–52% slower than dequant + cuBLAS.
  2. PXQ4 dp4a MMQ prefill tile: −49%.
  3. Wide-store dequant kernel: ~2× slower (294–322 vs 604–644 GB/s).
  4. Side-stream dequant prefetch arena: correct, does not pay (stays 0).
  5. Cross-ubatch device pipelining: −6.3% / −4.3% at `-ub 2048`.
  6. Async input staging: −18%, and a corruption race.
  7. Scheduler copy depth 4: correct, "not the lever".
  8. Explicit `-ts` layer rebalance: −4.4%, −7.1%, −12.7%.
  9. Pascal fused norm + SwiGLU (vLLM sidecar): +13.4% decode but crashes on ~6.5k-token prompts.
  10. DeltaNet out-gate fusion bit: 1.6–3.7 t/s, left off after a race.
- **Overlap with our levers:**
  - **CUDA graphs on Pascal** (not among the ten, documented elsewhere): keyed graph replay −3.9% on P100
    (`docs/DELTA-SINCE-IK.md:99-103`); graph decode slower and aborts on the second request
    (`docs/KNOWN-ISSUES.md:622-633`); tensor-split capture + replay 33.44 vs 35.40 t/s eager on 2× P100
    (`reduce.cu:546-548`). **Opposite sign to our +32%.** PXA's decode is GPU-bound with one launch per reduce; ours
    was host-bound under the meta butterfly's ~33 host calls per exchange. Expect our graph gain to shrink once the
    exchange is cheap.
  - **NCCL:** PXA measured NCCL's SHM route at 2.9 ms per 5120×512 f16 reduce (1.8 GB/s), "a quarter of the prefill
    wall on a P100 pair", and replaced it with its own push (`reduce.cu:1986-1991`). ik's own comment: NCCL gives
    "suboptimal prompt processing performance when we have more than 2 GPUs". Consistent with our SHM-vs-P2P gap.
  - **Batch/ubatch:** PXA's 2× P100 dense default is `-ub 256` ("340 t/s at 3.1k with `-ub 256`, against 231 with
    2048", `docs/DEFAULTS.md:42`), 4× P100 dense `-b 2048 -ub 256`. **Opposite to our sweep** (2048 best), on a
    different engine and exchange; we did not test 256.
  - **Layer-split pipelining / KV rebalancing:** negatives 5–8 say pipelining across ubatches lost on their hybrid
    model and `-ts` rebalancing was a monotone loss.
  - **MTP:** "Pascal speculation waits on an sm_60 token-folded verify kernel" (verify at M=8 is 7.0× a decode step,
    `RELEASE-NOTES-2026-09-07.md:643-656`); by v2026.10 MTP is auto on a multi-card tensor split, depth 3 with a
    top-1 floor of 0.8 on P100, depth 2 with `--spec-type mtp` (`docs/DEFAULTS.md:43-44`).
  - **mmap:** only one mention (a per-layer embedding table mmapped from a spinning disk stalls the first long
    prompt). Nothing on mmap noise.

**3. Multi-slot fairness (463 s → 2.7 s).** Two server changes in
[`server-context.cpp`](https://github.com/poisonxa16/pxa/blob/232fd83/examples/server/server-context.cpp): `PXA_PROMPT_FAIR_v1` (lines 6078-6115) gives the one
prompt seat per batch to a pending slot whose remaining prompt fits in a single batch, instead of the lowest slot id;
`PXA_INTERLEAVE_DECODE_MS_v1` (lines 7884-7895, 7965-7974) runs decode ticks for other slots for up to a second after
each prompt chunk. Result (`RELEASE-NOTES-2026-09-07.md:276-294`): a 23-token chat during a 100k prefill, first token
2.70 s vs 463.2 s; the prefill loses nothing measurable. Residual: one chunk time (~19 s for 2,048 tokens at 145k).
- **Applies to the Kmic fork only with ≥2 slots.** The fork serves `-np 1`; with one slot there is nothing to
  interleave and a second request simply queues. With `-np 2 --kv-unified` (the flag exists, `common/arg.cpp:1721`)
  the same two changes (~100 lines, MIT) would port to the fork's upstream-derived loop
  (`tools/server/server-context.cpp:3202-3230`).
- **Catch:** the fork's tick is `-b`-sized, and the fork needs `-b 32768` for MTP at long context. A tick of up to
  32,768 tokens is ~2 minutes at 275 t/s, so the prompt would also have to be chunked below `-b` for the decode
  window to matter. PXA's seat uses `-b 2048`.

**4. Kernel changes: portable vs PXQ-bound.**
- **Portable under MIT (in-tree, format-independent):** the all-reduce routes and their slot-safety and
  graph-capture arguments (`reduce.cu`); the server fairness changes; the elementwise-chain fuser
  (`PXA_EW_FUSE`); flash-attention work (tile-f16 mask skip, D=256 GQA-packed decode, quantized-KV direct read,
  sm_60 decode kept on `vec_f32` while prefill stays fp16); delta-net fusions (note the two aliasing races they had
  to fix). **These port as algorithms, not drop-ins:** PXA is ik_llama-based, the Kmic fork is upstream-based.
- **PXQ-bound:** the matvec wins (fused up+gate MMVQ, the K8-2D split decode mmv at +35.1% / +14.7%, int8 prefill
  tile, dequant kernels) are kernels for PXQ slab layouts; they are in-tree MIT but only help PXQ files.
- **Closed:** every PXQN kernel ships as `libggml-pxqn.so` (`ggml/src/ggml-pxqn-api.h:1-8`), including the PXQN form
  of the tensor-split epilogue.
- **On k-quants the engine itself is worth little for decode:** same-quant decode vs upstream ik is +2.7–3.3%
  (`bench/fair-battle.md`, same-quant control). Nothing in PXA speeds up Q6_K or Q8_0 matvec on Pascal.
- I did not audit each kernel file line by line for format dependence.

**5. Converting Qwen3.8-27B to PXQ4 or PXQU.**
- **Needs:** a Q8_0 GGUF source (the unsloth Q8_0 with the MTP head qualifies) and `pxq-quantize`, which is **not in
  the source tree** (`docs/QUANTIZING.md:21`, `docs/tutorials/04-quantize-your-own-model.md:19-25`; a separate
  download; the README's supporter tier lists "the quantizer key", so access may be gated — not verified).
  Command: `pxq-quantize --allow-requantize q8.gguf out.gguf PXQ4`. No imatrix (ignored by design). `ssm_out` and
  the MTP companion block stay q8_0 (`docs/QUANTIZING.md:132-140, 176`).
- **PXQU** is a per-tensor tier map (`--pxq-universal map.tiers`, `docs/PXQU-CONVERT.md`) aimed at MoE experts; for
  a dense 27B the relevant tiers are PXQ4 / PXQ4-HQ / PXQ6, or PXQN4 / PXQN5 (closed codec).
- **The result runs only in PXA** (PXQ is CUDA-only; PXQN needs the closed library). The Kmic fork cannot load it.
- **Reported quality**, KLD vs Q8_0 on 15,065 assistant tokens of a chat set (README table; Q8_0 is the reference,
  so no Q8_0 row):

  | file | size | KLD vs Q8_0 | same top token |
  |---|---:|---:|---:|
  | Q6_K | 22.4 GB | 0.0023 | 98.3% |
  | PXQN5 | 18.8 GB | 0.0022 | 98.4% |
  | PXQ6 (classic) | 18.8 GB | 0.0076 | — |
  | PXQN4 | 15.7 GB | 0.0079 | 97.1% |
  | PXQ4-HQ (classic) | 16.5 GB | 0.0153 | — |
  | PXQ4U (classic) | 15.7 GB | 0.0192 | — |

  PXQ4-class files are 3.4–8× further from Q8_0 than Q6_K; PXQN5 matches Q6_K at 84% of the size. Their metric
  and corpus differ from ours, so these are not comparable with our NCCL KLD numbers.

**6. Benchmarks against the Kmic fork.** Never named. `RELEASE-NOTES-v2026.10.md:19-33` compares against "the fastest
other Pascal build we tested, Q6_K", file size **22,431,001,568 B** — byte-for-byte the size of our own pure-Q6_K
requantization of Qwen3.8-27B with the MTP head — and that build has "its own MTP drafting". **Inferred, not stated:
this is the Kmic fork.** That table is the closest like-for-like cell:

| 2× P100, `-sm tensor`, pure Q6_K (PXA's measurement of the other build) | t/s |
|---|---:|
| tg128 / tg256 at 16k | 32.54 / 31.53 |
| pp512 / pp4096 / pp16384 | 273.67 / 272.10 / 267.15 |
| MTP, server first request, prose / code | 45.92 / 64.00 |

What differs from our numbers: 2 cards vs our 3; pure Q6_K vs our XL; `-b 2048 -ub 512 -t 8` vs our
`-b 32768 -ub 2048`; tg128 vs tg512; their PCIe link width for the pair is not stated in what I read (the 4-card
rig is x4; the test machine is x8); their MTP is a server request with that build's own drafting defaults vs our
`llama-speculative-simple` greedy runs. `bench/fair-battle.md` is against upstream ik_llama.cpp on single cards
with a 35B MoE, not relevant here.

**7. Does this change "nobody has done the 3-GPU work"? Partly.**
- **Still true:** nobody has published a 3-GPU (odd N) Pascal tensor split. PXA says in three places that 3 cards
  are unmeasured and routes them to layer split.
- **No longer true:** "no N-way, Pascal, PCIe-only P2P all-reduce validated on 3+ real GPUs". PXA runs a tensor
  split on **four real P100s over PCIe x4** by default, with P2P reduces, and ships a one-kernel all-reduce that
  runs on sm_60. Its N-way fused route exists but is off by default; 4-card decode goes through NCCL (SHM) or the
  in-tree peer route, and 4-card prefill through ik's ring.
- **Consequences for this plan:**
  - PXA's `reduce.cu` is a better porting source than ik_llama's: already adapted to Pascal (no `__nanosleep`, no
    peer atomics), with a written slot-safety argument and a device-side token counter that survives CUDA graph
    capture.
  - PXA's own numbers match ours: a decode reduce costs "47-67 us of GPU span, x130 per token — about 30% of a split
    token" on their pair (`reduce.cu:417-419`); our replicated meta pattern measured 49.4 µs × 128.
  - Their 4-card decode is **slower** than their 2-card decode (30.7 vs 37.8 t/s), the same shape as our 3 < 2.
  - Their design is pull + device staging; ours (Phase 1.5 shootout) is push. Both worked on PHB topologies.

**Follow-up measurements on the test machine (not run; listed for a later phase):**
1. Add PXA's pull-with-device-staging all-reduce to the out-of-tree shootout as a fifth arm and compare with one-shot
   push at 20 KB and 64 KB (latency, p99, bit-identity at N=3 in a fixed rank order).
2. Prefill `-ub 256` and `-ub 512` at short prompts (3k) and at depth, on configs A and B. PXA finds `-ub 256` best
   on a P100 pair; our sweep stopped at 512 and used pp2048 only.
3. CUDA graphs `=3` on vs off **with NCCL** (config B) for tg512 and MTP: PXA finds graphs a loss once decode is
   GPU-bound, and NCCL already removes most of the host calls.
4. `/dev/shm` size under NCCL in the Compose service (PXA's bug #206: a multi-rank SHM group fails in Docker's 64 MB
   default). Our config forces P2P, but a fallback to SHM would hit this; test with `shm_size: 1g` and without.
5. Multi-slot behaviour of the fork's server: `-np 2 --kv-unified`, a short chat during a 65k prefill, time to first
   token, with `-b 32768` and with `-b 2048`.
6. MTP under sampling (temp 1.0, the fork's `LLAMA_SPEC_*` settings): p-min 0.0 vs a confidence floor near PXA's
   0.8, n-max 2/3/4.
7. An epilogue-fusion estimate from the item 1 budget: norms (1.9 ms/token) and exchange adds are the kernels PXA
   folds into its reduce.
8. A cross-engine reference, only if you want it: the PXA release on the test machine with the pure Q6_K at 2 cards
   (tensor) and 3 cards (layer; tensor is refused or needs `PXA_TSPLIT_REDUCE_NWAY=1` and an explicit `-sm tensor`).

## 2. Code analysis of the fork

### 2.1 Every place that assumes exactly 2 devices or a 2-GPU topology

| Location | Assumption | Effect on 3 GPUs |
|---|---|---|
| `ggml/src/ggml-cuda/allreduce.cu:402` | `if (n_devices != 2) return nullptr` | Never runs; also fails `cc < VOLTA` at `:412` on any P100 |
| `allreduce.cu:607`, `:755` | `GGML_ASSERT(n == 2)` | Unreachable after the init guard |
| `allreduce.cu:618-661`, `:909` | `cuda_ctx[2]`, `peer = 1 - i` ("valid for n == 2 only") | Buffer and sync layout is pairwise |
| `allreduce.cu:228-234` | `GGML_CUDA_AR_POOL_SIZE = 2`, justified by "the two GPUs are at most one AR apart" | Ring-depth reasoning is pairwise |
| `ggml-cuda.cu:1306-1315` | `ggml_cuda_ar_p2p_state` with `ready[2]` and `done[2]`, a single process-wide `static` instance, `backends.size() != 2 → false` | The fork's P2P AR is off; one static state also rules out several pairs (grouped TP2) |
| `ggml-cuda.cu:1340` | `ctx[0]->xchg_peer = ctx[1]` (one peer pointer) | No overlapped prefill exchange on N≠2 |
| `ggml-cuda.cu:1362` | ≤256 KB → one-kernel path; larger → `ggml_cuda_allreduce_chunked` | Both are 2-GPU only |
| `ggml-cuda.cu:1199` `k_ar_p2p_f32(t0, t1, lo, hi)` and `:1376` `lo = j == 0 ? 0 : half` | Two pointers, two halves; GPU j **reads** the peer's half over P2P | Needs N pointers and N slices; reads are a risk on Sandy Bridge-E |
| `ggml-cuda.cu:1222` `ggml_cuda_allreduce_chunked(ctx[2], …)` | `1 - j` throughout; one landing buffer per device | Each device needs N−1 landing buffers for a one-shot |
| `common.cuh:1526-1564` | `peer_stage[2]` (one OUT, one IN), `peer_stage_free` single event, `xchg_peer` "the other GPU of a two-GPU tensor split" | The butterfly stays correct (stages are serialized), but N-way concurrency needs per-peer buffers and events |
| `gemm-fold.cu:751, 829` and `ggml-cuda.cu:5351` | `xchg_want` is set for the last node of **every** subgraph regardless of N, so the fold GEMM splits into 4 token chunks with events, and gate/up pairing is skipped for that node (`gemm-fold.cu:777` `!ctx.xchg_want`) | **[inferred]** Pure overhead on N≠2: the chunks are never consumed (`xchg_peer` is null). A cheap fix candidate |
| `ggml-backend-meta.cpp:2421-2475` | Butterfly; for N not a power of 2: fold the excess (2→0), butterfly (0↔1), copy back (0→2); `n_reduce_steps = ceil(log2 N)` (`:1842`) | **Three dependent stages per exchange on N=3** (vs one for the fork's 2-GPU kernel); GPU1 idles in stage 1, GPU2 in stage 2 |
| `ggml/include/ggml-backend.h:364` | `GGML_BACKEND_META_MAX_DEVICES 4` (the fork shrank it from 16; CHANGES §10 `a8b274ea6`) | 3 is fine; **caps "3 or more" at 4 GPUs** |
| `ggml-cuda.cu:408-421` | Enables peer access between **all** pairs when `GGML_CUDA_P2P` is set | N-generic, OK |
| `ggml-cuda.cu:2910-2955` | Peer copy f16 compression after a one-time exactness probe; copies issued on the **source** device's copy stream | N-generic. A source-issued `cudaMemcpyPeerAsync` is a push, which is good on Sandy Bridge-E **[inferred]** |
| `ggml-cuda.cu:1068` (NCCL path) | f32 for small tensors (N=3: below 131072 elements), **BF16 above** | NCCL on N=3 is lossy for prefill as written; the fork rejected BF16 wire for exactly this reason (OPTLOG 189) |
| `ggml-backend-meta.cpp` graph compute loop | One host thread enqueues each device's subgraph in turn (`HANDOFF.md` open thread #1: ~200 µs skew per graph on 2 GPUs) | **[inferred]** Skew and enqueue cost grow with N, especially with 3-stage exchanges |

Other synchronization notes: the fork's peer copy waits on the destination's `work_event` to avoid a
reuse race (`ggml-cuda.cu:2976-2981`). It shipped two real cross-stream races before (CHANGES §6), so
any new N-way path must go through the race-stress tests (section 7).

### 2.2 How Qwen3.8-27B is split across 3 GPUs (`src/llama-model.cpp:696-851`)

Model facts read from the GGUF header on Hugging Face (`unsloth/Qwen3.8-27B-GGUF`, HTTP range request, no
local file touched): `qwen35` arch, 65 blocks (17 full attention at `full_attention_interval 4`, 48 gated
delta-net; block 64 carries the MTP `nextn` layer), embedding 5120, FFN 17408, 24 Q heads, **4 KV heads**,
head size 256 (GQA 6), delta-net state 128 with 16 groups and 48 value heads.

Granularity rules **[code]** and the resulting `-ts 1/1/1` split **[inferred from code]**:

| Tensor class | Granularity | Units | Split on 3 GPUs | Split on 2 GPUs |
|---|---|---|---|---|
| Full-attention Q (with Qwen Q-gate) | lcm(2·1536, blck) = 3072 | 4 KV groups | **1/1/2** (rotated per layer) | 2/2 |
| K, V, KV cache | 256 = 1 KV head | 4 | **1/1/2** | 2/2 |
| attn_output (input axis) | 1536 | 4 | **1/1/2** | 2/2 |
| FFN up/gate/down | lcm(blck, 128): 256 for K-quants, 128 for Q8_0 | 68 or 136 | 22/23/23 or 45/45/46 | even |
| Delta-net qkv (k segment, 16 heads × 128) | 256 or 128 | 8 or 16 | 2/3/3 or 5/5/6 | even |
| Delta-net v (48 heads × 128) | 256 or 128 | 24 or 48 | 8/8/8 or 16/16/16 | even |

The rotation (`rotation = get_il_eff(il) % n_devices`, `:459`) moves the extra KV group to a different GPU
on each full-attention layer. Averaged over layers the load is balanced, but **within each full-attention
layer one GPU holds half the attention work and half that layer's KV cache, and every per-layer exchange
waits for it** **[inferred]**. The fork notes flash attention is "the only cost on this model that scales
with context length" (OPTLOG "measured dead ends" section), so this imbalance matters most for long-prompt
prefill and deep-context decode. `-ts` cannot fix it, because the granularity forces whole KV groups.

Related **[measured]** observation from the round-4 battery telemetry: GPU0 averaged 80.7% utilization
against 52.8% and 53.2% for GPU1 and GPU2. GPU0 does something extra or is the device the others wait
on; Phase 1 attributes it.

### 2.3 Shape- and quant-specific kernel tuning, and how much applies to UD-Q6_K_XL

Shape-specific **[code/docs]**: flash-attention tile configs assume head size 256 with GQA 6 (FINDINGS
"What doesn't transfer"); MTP verify width 5 (n_max 4) sets the fixed-width verify graph; the mmvq Pascal
launch geometry is tuned on a Q6_K model and gated on `__CUDA_ARCH_LIST__ == 600`. All of these match
Qwen3.8-27B, so they apply on the test machine.

Quant-specific **[code]**: `mmvq-f16.cu` (the fp16 verify matvec and the fused FFN gate+up+SwiGLU, OPTLOG
231) requires `src0->type == GGML_TYPE_Q6_K` (`mmvq-f16.cu:456`). The Q6_K `vdr`/staging work (CHANGES §1)
targets Q6_K. Q8_0 and Q5_K get the shared Pascal mmvq geometry, with some per-type rows, but not the
Q6_K-specific kernels. mewsian's PR #1 shows per-type retuning is real work: "the shared one was tuned on a
Q6_K model".

#### 2.3.1 The model file behind the fork's published numbers

**What the fork says [code/docs]:** `tools/gate.sh` defaults to
`MODEL=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf`; `QUICKSTART.md`, `BUILD.md`, `bundle-README.md`, the
`p100-handoff/bench/*.sh` scripts and the OPTLOG header all use that same local filename
(`/path/to/Qwen3.8-27B-Q6_K.gguf` in the user-facing docs). **No source repo, download command or
quantize command appears anywhere** in the fork's docs, scripts or commit messages.

**Fingerprints in the fork's own logs [code/docs]:**
- "Model is 20.88 GiB, tensor-split -> 10.44 GiB of weights read per GPU per token" (`OPTLOG.md:4321`;
  also `:3195`, where the same Q6_K is compared with a 14.94 GiB `Qwen3.8-27B-Q4_0.gguf`).
- "`output.weight` is 5120x248320 Q6_K = **995 MiB**" (`OPTLOG.md:2056`).
- The delta-net alpha/beta projections run as **q6_K** matvecs, "5120x24 per GPU … 96 per pass"
  (`OPTLOG.md:7417-7421`; CHANGES `04e9262d1`).

**Matching against Hugging Face [measured, GGUF headers fetched with HTTP range requests]:**

| Candidate | Tensor bytes | `output.weight` | `ssm_alpha`/`ssm_beta` | Q8_0 tensors | Match? |
|---|---:|---|---|---|---|
| **Pure Q6_K** (every quantized tensor of this model in Q6_K; computed from the tensor list) | **20.880 GiB** | Q6_K, 994.6 MiB | Q6_K | none | **Yes, all three fingerprints** |
| unsloth `Qwen3.8-27B-Q6_K.gguf`, **deleted** from `unsloth/Qwen3.8-27B-GGUF` on 2026-08-19 (commit `24fde40268`; parsed at revision `db81afd1e1`) | 21.303 GiB | Q6_K | **F32** | 49 (48× `ssm_out`, `nextn.eh_proj`) | No (size, alpha/beta) |
| unsloth `Qwen3.8-27B-UD-Q6_K.gguf` | 20.464 GiB | **Q8_0** | — | 154 | No |
| bartowski `Qwen3.8-27B-GGUF/Qwen3.8-27B-Q6_K.gguf` | 22.212 GiB | Q6_K | — | 5.98 GiB | No |
| ggml-org `Qwen3.8-27B-GGUF` | no Q6_K (BF16, Q8_0, Q4_K_M only) | — | — | — | — |

**Conclusion:** the fork measured on a **pure Q6_K** (all 2-D weights in Q6_K, only norms and small
vectors in F32) that matches no file published on Hugging Face now or (for unsloth) before
2026-08-19. It was very likely made locally with plain `llama-quantize … Q6_K`. **[inferred]** The
exact source repo is **not determinable** from the fork. Its other file, `Qwen3.8-27B-Q4_0.gguf`
(14.94 GiB), matches unsloth's file of that name exactly (14.944 GiB), so the author does download from
unsloth, and the BF16 source may also be unsloth's. That is not confirmed. To reproduce the reference:
`llama-quantize` a BF16 (`ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-BF16.gguf`, 50.1 GiB, or unsloth's
BF16) to Q6_K and check the result is 20.88 GiB with Q6_K `output.weight` and `ssm_alpha`/`ssm_beta`.

#### 2.3.2 What UD-Q6_K_XL has that the pure Q6_K does not

The pure Q6_K contains only **Q6_K** and **F32**. UD-Q6_K_XL adds three tensor types:

| Type in XL, absent from pure Q6_K | Tensors | Bytes | Where |
|---|---:|---:|---|
| **Q8_0** | 310 | 12.60 GiB | `output.weight` (1.26 GiB, read every token); FFN up 2.82, down 2.56, gate 2.03 GiB (whole layers 55-63, scattered elsewhere); `attn_qkv` 1.14; `ssm_out` 1.03; `attn_gate` 0.81; `attn_q` 0.37; `attn_output` 0.40; `attn_v`/`attn_k`; all 48 `ssm_alpha` and `ssm_beta` |
| **Q5_K** | 27 | 1.00 GiB | scattered FFN, `attn_qkv`, `attn_gate`, `ssm_out`, `attn_q`, `attn_k`, `attn_v` |
| **Q4_K** | 1 | 0.016 GiB | one `attn_gate` |

**Per-token weight stream** (excluding the `token_embd` lookup, which reads one row, and the MTP block
`blk.64`, which only the draft uses):

| File | Stream per token | Q6_K | Q8_0 | Q5_K | Q4_K |
|---|---:|---:|---:|---:|---:|
| Pure Q6_K (fork reference) | ~19.6 GiB | 100% | — | — | — |
| **UD-Q6_K_XL** (the test machine) | **22.24 GiB (+13.6%)** | **38.9%** | **56.6%** | 4.5% | 0.1% |

**How much of the fork's kernel tuning applies to the XL [code + measured shares]:**
- **Q6_K mmvq work** (CHANGES §1: `vdr` 4, q8_1 activation staging, integer accumulation, vectorized
  Q6_K dequant, 1-row blocks for small q6_K matrices): applies to **38.9%** of the XL's per-token bytes,
  versus 100% on the reference. Some of §1 is type-generic (activation staging, geometry): **[inferred]**
  partial benefit for Q8_0 and Q5_K.
- **fp16 verify matvec** (`mmvq-f16.cu`, Q6_K only, `mmvq-f16.cu:456`): the MTP verify pass, which is
  "almost all" of the 56.8 ms verify in a ~72 ms cycle on the reference (CHANGES §10). It applies to the same **38.9%**.
- **Fused FFN gate+up+SwiGLU** (OPTLOG 231, +2.1% verify on the reference): needs both `ffn_gate` and
  `ffn_up` in Q6_K, which is **21 of 64 target layers (12.9% of bytes)** versus all 64 on the reference.
- **Small q6_K matrix path** (alpha/beta 5120×24 per GPU, 96 calls per pass): **0%**. They are Q8_0 in the XL.
- **Output head** (`output.weight`, 1.26 GiB per token, split across GPUs): Q6_K on the reference, **Q8_0 in the XL**.
- **Type-independent work applies fully**: flash-attention tile and q4_0 KV dequant (§2), GEMM attention
  and fold GEMMs for prefill (§3, §10-14, which dequantize to fp16 regardless of quant type), cuBLAS
  algorithm choice, CUDA graphs, MTP drafting and verify logic, the compact mask, and the
  tensor-parallel copy work.

So **[inferred]** the decode and verify tuning applies to roughly 2/5 of the XL's weight stream; the
prefill and attention tuning applies fully. Before its Q6_K work the fork measured untuned mmvq at
**332 GB/s for q8_0 versus 208 GB/s for q6_K** (605 GB/s ceiling; OPTLOG "Diagnosis"), so untuned Q8_0 is
not slow per byte. The open question is whether the XL's Q8_0 matvecs approach the tuned Q6_K kernels per
weight. Phase 1 G3 measures it, and the 2-GPU control in G1 should use a **pure Q6_K** (reproduced as
above) so that the test machine can be compared like for like with the fork's published numbers.

### 2.4 A rough decode budget [inferred, to be replaced by Phase 1 data]

- The test machine, 3 GPUs, XL, `-sm tensor`: tg256 = 20.95 ± 3.85 t/s from `gate.sh` → **47.7 ms/token**.
- Weight bytes streamed per token (section 2.3.2): 22.24 GiB for the XL, which is 7.96 GB per GPU on 3 cards. At the
  fork's measured 605 GB/s streaming ceiling that is a **~13.2 ms floor**.
- The fork's 2-GPU pure-Q6_K box: 32.6 t/s → 30.7 ms. Its stream is ~19.6 GiB, 10.5 GB per GPU (the fork's own
  figure, counting the whole file, is 10.44 GiB per GPU; OPTLOG 4321). That is a ~17.4 ms floor, so about
  57% floor efficiency.
- If the test machine reached the same efficiency it would run at about 23 ms/token (about 43 t/s). **About 24 ms
  per token is unexplained.** Candidates: the 3-stage exchanges (~128 per token), host enqueue for 3
  devices on a 2011 CPU, kernel efficiency on Q8_0, P2P platform latency, and the 2/1/1 attention
  imbalance at depth. Phase 1 splits this up.
- **Pure Q8_0 arm (unsloth `Qwen3.8-27B-Q8_0.gguf`, section 6) [inferred]:** the per-token stream is 25.35 GiB
  (27.22 GB; +14.0% vs the XL, +29% vs the pure Q6_K), which is 9.07 GB per GPU on 3 cards. That gives a
  **15.0 ms floor**. At the fork's 57% efficiency it would be ~26.3 ms/token (~38 t/s); at the test machine's current
  XL efficiency (13.2 / 47.7 = 28%) it would be ~54 ms/token (~18.5 t/s). The fork's untuned q8_0 mmvq rate (332 GB/s,
  OPTLOG "Diagnosis") puts matvec time alone at ~27 ms/token. On 2 cards it is a 22.5 ms floor, and 13.5 GiB of
  weights per card may not fit at useful context.
- **Why report ms, not just t/s:** exchange and host costs are roughly fixed per token whatever the quant
  **[inferred]**, so the same absolute saving looks smaller in percent on a heavier quant. Saving `X` ms from a
  `T` ms token gives `X / (T − X)` more t/s. For example, 5 ms is +11.7% at the XL's 47.7 ms but only +10.2% at
  Q8_0's ~54 ms. Every phase below therefore reports both.

Prefill comparison (different tools, indicative only): the fork's own server table on 2 GPUs shows 423 t/s
at 32k and 342 t/s at 62k (QUICKSTART). the test machine's round-4 server log shows 257 t/s at about 19k and 218 t/s
at 64k on 3 GPUs.

## 3. Test machine topology (measured 2026-10-02, read-only)

- `nvidia-smi topo -m`: every GPU pair is **PHB**; one NUMA node; CPU affinity 0-11.
- `nvidia-smi topo -p2p r` and `-p2p w`: **OK for all six directed pairs**.
- PCIe: **all three P100s at Gen3 x8** (capable of x16), not two. Both IIO root-port groups are
  bifurcated x8/x8: GPU0 is on `00:02.0` (sibling `00:02.2` is empty, link down); GPU1 and GPU2 are on
  `00:03.0` and `00:03.2` (one bifurcated x16 port). This matches `HARDWARE.md` and `BIOS-FIX.md` in
  the machine's own notes (riser removed 2026-09-29; IOU2 and IOU3 set to x8x8).
- CPU: i7-3930K (Sandy Bridge-E, 2011), 1 socket, integrated IIO root complex. IOMMU is off (no
  `intel_iommu` on the kernel command line, 0 IOMMU groups), so ACS will not force P2P through the root
  complex.
- Driver 580.178.04. A GT 610 display card sits on `00:01.0` (not a CUDA device for this driver).

What this decides:
- P2P is possible between **every** pair, so designs (a) and (c') are not ruled out.
- **[inferred, must measure]** "OK" doesn't mean fast. Sandy Bridge Xeon is documented at about 800 MB/s
  for P2P **reads** (source 14). The driver may also stage P2P through system memory on some
  topologies (source 9). Pull-based designs (the fork's `k_ar_p2p_f32`, ik_llama's one-shot) are at risk
  here; push designs (P2P stores, or `cudaMemcpyPeerAsync` issued on the source) less so. Phase 1 G0
  measures read and write bandwidth and latency per pair.
- **[inferred]** GPU1↔GPU2 share one IIO port (bifurcated `00:03.x`) and may have better P2P than either
  pair involving GPU0 (`00:02.0`). If so, that is the natural pair for any 2-GPU design.
- Optional, your call: GPU0's port could go back to x16 (`IOU2 = x16`), since `00:02.2` is empty.
  GPU-SCALING.md found link width irrelevant for layer split; for tensor-split prefill (MB-scale
  exchanges) it's unknown. Low priority.

## 4. Candidate designs, ranked

Expected gains are **[inferred]** ranges until Phase 1 provides the exchange share `s` and the P2P numbers.
The general rule: if a design removes fraction `r` of time share `s`, the speedup is `1 / (1 − r·s)`.

| Rank | Design | Expected gain | Effort | Risk | What to measure first | P2P on all pairs? |
|---|---|---|---|---|---|---|
| 1 | **(d) Measure, then cheap levers on the current fallback:** (i) an NCCL build (`libnccl` 2.27.3 and `nccl.h` are already in the build image); decode stays f32 and exact, but prefill needs a patch to keep f32 or f16-exact instead of BF16; (ii) gate `xchg_want` on `xchg_peer != nullptr`; (iii) `-sm layer` vs `-sm tensor` decision; (iv) quant choice (pure Q6_K vs UD-Q6_K_XL) with a quality check | 0-30% combined; quant alone could be large for decode | Hours to 2 days | Low (env toggles, rebuilds) | NCCL Pascal init and small-message latency; tg/pp per quant | NCCL: decides its own transport |
| 2 | **(a) N-way P2P AllReduce in the fork's comm path.** Decode (≤256 KB): a **one-shot push**, where each GPU writes its full partial into each peer's landing slot, then flag, then a local sum of N in a fixed order (bit-identical across GPUs, as the fork's 2-GPU kernel is). Prefill: **reduce-scatter + all-gather (two-shot) with token-chunk overlap and f16 wire**, generalizing `ggml_cuda_allreduce_chunked`. Algorithm choice for N=3: one-shot = 1 sync round, 2·n bytes written per GPU; two-shot = 2 rounds, 4n/3 bytes; ring = 4 rounds, 4n/3 bytes; tree = 2 rounds with a hot root. For decode (20-100 KB) latency dominates, so one-shot. For prefill (20-40 MB per exchange) bandwidth dominates and two-shot wins. | If `s_dec` ≈ 25% and `r` ≈ 0.7: about +20% decode. Prefill likely +5-15% unless the chunk overlap hides more | Decode 3-5 days; prefill 1-2 weeks | Medium: Pascal has no `__nanosleep` (spin like OPTLOG 189), P2P write performance on SNB-E, races | G0 P2P write and read per pair; G1/G2 exchange share | **Yes**, for all three pairs |
| 3 | **(e1) Q8_0 kernel parity:** port the fp16 verify matvec and fused gate/up to Q8_0 and mixed Q6_K/Q8_0 pairs; tune Pascal geometry for Q8_0 and Q5_K (as PR #1 did for Q4_K) | Depends on G3; covers the 56.6% of the XL's per-token stream in Q8_0 (plus 4.5% Q5_K); helps 2- and 3-GPU alike | 1-2 weeks | Medium (bit-exactness vs reference kernels; the fork's mmvq-harness helps) | Per-type mmvq time at the model's shapes (G3) | No |
| 4 | **(e2) KV-group rebalancing:** per-class split ratios in `llama_meta_device_get_split_state` so the GPU holding 2 KV groups in a layer gets fewer FFN or delta-net rows | Long context only; up to ~1/3 of the attention straggler time at 64k+ | 3-5 days | Medium (split invariants; meta backend assertions) | Per-GPU FA time per layer at -d 65536 (G2) | No |
| 5 | **(e3) Per-device enqueue threads** (fork HANDOFF open thread #1) | Fork estimates 2-4% at 2 GPUs; likely more at 3 | 3-5 days | Medium (cross-thread event ordering) | Host timeline (`LLAMA_TL=1`), enqueue skew per graph | No |
| 6 | **(c') Grouped TP2:** pairs per layer group (krampenschiesser-style `--max-tensor-split 2`) so every exchange uses the fork's 2-GPU kernel and chunked prefill, with all 3 cards' VRAM. Needs the `static` single-pair P2P state made per-group. | Theoretical memory time per token is 1/2 of one GPU rather than 1/3, so it only wins if comm dominates. Gives 262k context headroom | 1 week | Medium | 2-GPU vs 3-GPU ratio on the test machine (G1) | Only within each pair |
| 7 | **(b) TP2 on two cards plus the third card for something else** (a second model, embeddings, vision). Moving the MTP draft to card 3 isn't worth it: the draft shares the target's weights, steps are ~10% of a cycle (OPTLOG 189), and it would duplicate ~2.5 GB of embeddings and output. | Decode ≈ the test machine's 2-GPU number; no 3-GPU speedup; possible capacity win | Config only | Low; the XL fits 2×16 GB at about -c 65536, and 262k is tight **[inferred]** | 2-GPU numbers on the test machine (G1) | One pair |
| 8 | **(c) TP2 plus pipeline to the third card** | For single-stream decode, worse than TP2: memory time ≈ 1/2·(2/3) + 1/3 = 2/3 of one GPU. Only helps throughput with several concurrent requests | 1 week | Medium | n/a | One pair |

`-sm layer` (all N): the fork measured it as a dead end on 2 GPUs (prefill 226.9, decode 20.4 t/s
against tensor ~31; OPTLOG "measured dead ends"). On 3 GPUs, single-stream layer split has a memory floor
of about one whole GPU (~40 ms/token) **[inferred]**. That is close to the test machine's current 47.7 ms, so it
needs measuring rather than assuming. It is part of G4.

## 5. Measurement gate (Phase 1, before any development)

All runs in the existing isolated Docker image (CUDA 12.9, `nvprof`/`ncu` present; host `nsys` is
2026.3, and its Pascal CUDA-trace support is unverified because the fork used 2022.4, so `nvprof` in the
container is the primary tool). Serving stopped, 125 W cap held, nothing in the model directory touched
until its download finishes.

**G0: P2P characterization (~1 h).** Build `p2pBandwidthLatencyTest` (cuda-samples, CUDA 12.9 tag) in the
container, plus a ~100-line custom test: a kernel streaming remote **reads** versus remote **writes**
(16 B/thread) per directed pair, and a flag ping-pong latency test (P2P store, then a local volatile poll).
Report GB/s and µs per pair and direction, with P2P enabled versus disabled (staged).
*Kill criterion for pull designs:* remote-read bandwidth below 2 GB/s on any pair.
*Kill criterion for push designs:* remote-write below 3 GB/s or flag round-trip above 10 µs.

**G1: Decode attribution (~2 h).** `llama-bench -p 0 -n 512 -r 3` under `nvprof --print-gpu-trace` for:
3 GPUs `-sm tensor` with all three arms (the XL, the pure Q6_K and the pure Q8_0); 2 GPUs `-sm tensor`
(`CUDA_VISIBLE_DEVICES` pairs 1,2 and 0,1) with the XL and with the **pure Q6_K, the required arm** for the
2-GPU control (Q8_0 on 2 GPUs only if it fits at `-c 16384`). The pure Q8_0 serves as the **uniform-type
attribution base**: every matvec is the same kernel, so exchange, idle and host time can be separated without
the XL's per-layer type mix. Use the fork's differencing trick (OPTLOG 125: profile `-n N` and `-n 1`, subtract) to cancel
setup. Per GPU per token compute: matvec kernel time; exchange time (the meta ADD one-node graphs,
f16 narrow and widen kernels, `memcpyPeer`); and **idle gaps adjacent to exchanges**. Also run the
fork's host instrumentation (`LLAMA_TL=1`, `GGML_CUDA_OP_PROFILE=1`, `LLAMA_UBATCH_PROFILE=1`).
Output, per arm: ms/token, matvec ms, exchange ms, exchange-adjacent idle ms, host enqueue ms, and
`s_dec` = (exchange + exchange-adjacent idle) / token time.

**G2: Prefill attribution (~2 h).** `llama-bench -p 2048 -n 0 -d 16384` and `-d 65536` (cost of 2048 new
tokens at depth), same configurations including the pure Q8_0 arm on 3 GPUs. Output, per arm: ms per 2048-token
step, exchange ms, `s_pp`, and per-GPU flash-attention time per
full-attention layer to quantify the 2/1/1 straggler.

**G3: Quant and kernel attribution (~1 h).** The fork's `p100-handoff/tools/mmvq-harness` (about 7 s per
kernel) at the model's shapes for Q6_K, Q8_0 and Q5_K at n=1 and n=5. Plus tg512 for the XL vs the pure Q6_K vs the
pure Q8_0 on the same GPU count. Report ms/token and per-type matvec ms. Q8_0 versus pure Q6_K isolates the
per-weight cost of the two kernel paths; the XL should then land between them in proportion to its 56.6% / 38.9%
split, and any shortfall beyond that points at the mix itself (per-layer type switches, the fused path covering
only 21 layers).

**G4: Split mode.** The earlier layer-vs-tensor llama-bench plan, corrected to tg512 (section 6).

**The single most decisive comparison:** the test machine 2-GPU `-sm tensor` with the pure Q6_K versus the fork's
published 2-GPU numbers (tg256 ≈ 31-32.6, pp2048 ≈ 493).
- If the test machine's 2-GPU result is near the fork's, the platform is fine and the 3-GPU shortfall is
  N=3-specific (exchanges, imbalance, enqueue). Designs (a), (e2) and (e3) are justified by the data.
- If the test machine's 2-GPU result is also far below, the cause is the platform (CPU, P2P, PCIe) or the quant,
  and N-way AllReduce work will not fix it.

**Expected gain and the threshold.** Projected gain from (a) = `1 / (1 − r·s) − 1` with `r` ≈ 0.6-0.8
for a one-shot push replacing the 3-stage butterfly **[inferred]**.
- **Go** with (a): `s_dec ≥ 20%` or `s_pp(64k) ≥ 20%`, **and** G0 push criteria pass on all pairs.
  At s = 20% and r = 0.7 the gain is about +16%.
- **Defer**: 10-20%. Try NCCL and the cheap levers first and re-measure.
- **Stop** (a): below 10% (≤ ~7% possible gain). Pivot to (e1), (e2) and (e3).

## 6. Benchmark methodology

Reuse `p100-handoff/bench/*` and `p100-handoff/tools/MEASURE.md`, with these fixes:
- **tg512 or longer**, never tg128. MEASURE.md: `-n 128` "amortises a ~2-3 s fixed startup … a 1.8x
  error". `gate.sh`'s own tg256 stays as the correctness-gate metric only.
- **Cool cards before every measured run**: poll until all GPUs read ≤ 45 °C, then one discarded warmup
  (QUICKSTART: hot P100s read "up to 20% low"; OPTLOG 4202-4205: first-of-batch 26.4 vs warm 30.1).
  Interleave A/B arms within one session ("cross-session comparison on this machine is worthless").
- **Identical flags across modes**: `-fa 1 -ctk q4_0 -ctv q4_0 -b 32768 -ub 2048 -ngl 99 -ts 1/1/1`
  (or `1/1` on 2 GPUs), with **fixed `-c`** (65536 for llama-bench). Only `-sm`, the GPU set or the model
  file changes. `GGML_CUDA_P2P=1` always. Record exact command lines and the build SHA.
- **3 reps, mean ± stdev** (llama-bench `-r 3` plus the warmup above). Report the tensor/layer ratio
  per prompt length.
- **Telemetry**: `nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,utilization.gpu
  --format=csv -l 1` bracketing every run; report per-GPU average utilization during pp16384 and
  pp65536 and the hottest temperature per run. Flag SM clock below 1000 MHz while busy, or anything
  above 72 °C.
- `llama-bench` has **no `-fit` flag**. Its fit logic is opt-in via `-fitt`/`-fitc`, so the
  `-sm tensor` fit hang only affects `llama-server`, which needs `-fit off`.

**Comparison matrix:**

| Axis | Values |
|---|---|
| Split | `-sm tensor`, `-sm layer` |
| GPUs | 3; 2 (pair 1,2 and pair 0,1) |
| Quant | UD-Q6_K_XL (current); **pure Q6_K** (the fork's reference, section 2.3.1), made with `llama-quantize` into a separate directory **after** the current model download finishes. From BF16: 50.1 GiB download + 20.9 GiB output, which doesn't fit beside the in-progress download in the ~65 GB free on `/` today. From ggml-org's Q8_0 with `--allow-requantize`: 26.6 + 20.9 GiB, same tensor types and sizes, so fine for **speed**, but slightly lossier, so not for quality claims. Optional third arm: unsloth UD-Q6_K (20.46 GiB). **Pure Q8_0 (uniform-type development and attribution base):** `unsloth/Qwen3.8-27B-GGUF`, file `Qwen3.8-27B-Q8_0.gguf` (29,047,086,048 bytes = 27.05 GiB on disk, 27.04 GiB of tensors; repo head `4ca720788d`, 2026-08-20). Its header shows 506 quantized tensors, all Q8_0 (including `output.weight`, `token_embd` and `ssm_alpha`/`ssm_beta`), 65 blocks with `nextn_predict_layers = 1`, and the `blk.64.nextn.*` tensors, so **it carries the MTP head** and works with `--spec-type draft-mtp` as a single file, the same shared-weight setup the fork measured. Its imatrix metadata doesn't matter, since Q8_0 quantization doesn't use one **[inferred]**. Not ggml-org's `Qwen3.8-27B-Q8_0.gguf` (26.62 GiB, the "~26.6 GiB" file): its header has 64 blocks and **no `nextn` tensors**. ggml-org ships the head separately as `mtp-Qwen3.8-27B-Q8_0.gguf` (2.94 GiB, with its own `token_embd` and `output`), which would have to be loaded as a separate draft model (`-md`), not the shared-weight MTP setup. The unsloth Q8_0 can also be the `--allow-requantize` source for the pure-Q6_K speed arm, keeping the MTP head. Disk: 29.0 GB, plus 22.4 GB for a requantized pure Q6_K, against ~65 GB free today before the in-progress download finishes. On 3 cards it is 9.0 GiB per card; on 2 cards, 13.5 GiB per card, which is likely too tight beyond short context **[inferred]**. |
| Tests | tg512; pp512, pp2048, pp16384, pp65536 (from empty); pp2048 at `-d 16384` and `-d 65536` |

The XL fits 2×16 GB at `-c 65536` (~11.8 GiB of weights per card plus ~0.6 GiB of q4_0 KV), but likely
not at 262k **[inferred]**.

**Agent-level battery at the end:** the same 9-task × 3-rep agent harness (fresh model-id, 125 W, the
server flags of the earlier battery plus `-fit off`), against the 09-27 baseline (96%, 145 s,
gen 45.1 t/s, MTP 59.5%) and the round-4 run (89%, 187 s, gen 44.6 t/s, pp avg 207.5 t/s, MTP 65.7%).
Given round 4's two `err_big_file_read` timeouts (slot-held 65k prefills), also report that task over 6+
reps.

## 7. Phased plan with gates

**Correctness gate for every change (from the fork):**
- `tools/gate.sh`: tg256 first on cold cards; perplexity on `p100-handoff/ppl-orig.txt` must stay within
  the band of the **same quant's** the test machine baseline (UD-Q6_K_XL: 2.6069 ± 0.0198 at round 4; the fork's
  2.6209 ± 0.0199 band belongs to its own Q6_K file); flash-attention op tests must pass.
- `tools/gate.sh --full` (~25 min, ~16k tests) for any comm or kernel change. The FA-only run "is not the
  full suite".
- `--kl-divergence` against the previous build at `-ub 5` (verify-shaped), max KLD reported. Exchanges
  that keep the summation order must be byte-identical.
- A 3-GPU long-context repeat-until-diverge check, given HANDOFF #6 (NaNs at 3 virtual devices on the GEMM
  attention path).
- A second-request server test (issue #29466 must not get worse).
- For new sync code: the fork's race-injection approach (`p100-handoff/wip-2026-09-14/code/
  temp-race-injection.patch`) adapted to N peers.

| Phase | Goal | Work | Time | Pass / fail | Decision |
|---|---|---|---|---|---|
| **0** | Plan | This document | done | — | **D0: you approve Phase 1** |
| **1** | Attribute the shortfall | G0-G4 (section 5); no source changes except an out-of-tree P2P microbenchmark; results in a local results file | ~1 day (6-8 h GPU) | All measurements captured with stdev; per arm (pure Q6_K, UD-Q6_K_XL, pure Q8_0): baseline t/s **and** ms/token (decode) and ms per 2048-token step (prefill at depth); `s_dec`, `s_pp` and exchange ms; P2P table; and the 2-GPU control on the pure Q6_K in hand | **D1: you pick which designs proceed** |
| **2** | Cheap levers | NCCL build (+ f32/f16 prefill exactness patch); `xchg_want` gating; quant decision with a quality check (KLD vs a Q8_0 reference, which fits on 3 cards); split-mode decision. Each behind an env toggle | 1-3 days | A kept item must show, interleaved A/B, a **t/s gain ≥ 3% on the arm it targets and ≥ 1.5 ms/token saved** (decode) or ≥ 3% / the matching ms per 2048-token step (prefill). Report both t/s % and ms saved on **all three arms**. Correctness gate passes | **D2** |
| **3** | N-way decode AllReduce | One-shot **push** kernel (or pull, if G0 shows reads are fine) in a new file (`ggml-cuda/allreduce-p2p-n.cu`) with minimal hooks in `try_allreduce_p2p`; per-peer landing buffers and events; Pascal spin instead of `__nanosleep`; the 2-GPU path must stay byte-identical | 3-5 days | Decode ms/token saved ≥ 60% of the Phase-1 projected exchange+idle ms; report **ms saved and t/s %** on all three arms. The ms saved should agree across arms within ±20%, since exchange cost doesn't depend on quant **[inferred]**; if it doesn't, find out why before D3. Full gate; race stress | **D3** |
| **4** | N-way prefill exchange and balance | Two-shot RS/AG with token-chunk overlap and f16 wire (generalize `ggml_cuda_allreduce_chunked`); optional KV-group rebalancing (e2); optional enqueue threads (e3) | 1-2 weeks | pp2048 at `-d 65536`: ms per 2048-token step saved ≥ 60% of the projected exchange+imbalance ms; report **ms saved and t/s %** on all three arms; decode ms/token unchanged or better. Full gate; long-context NaN check | **D4** |
| **5** | Integrate and decide | Agent battery, 6+ reps of `err_big_file_read`, soak run, second-request test; write up | 1 day | No ok% regression vs round 4; no new warnings or errors in the server log; report final gen t/s **and** ms/token, and prompt t/s and ms per 2048 tokens, against the round-4 baseline (44.6 t/s ≈ 22.4 ms/token gen avg) for the serving arm, plus the same pair for the other arms if they were run | **D5: you decide on serving** |

### Risks

| Risk | How it's handled |
|---|---|
| P2P slow or staged on this X79 board (Sandy Bridge-E reads ~800 MB/s class; the driver may stage via system memory) | Measured first (G0). Push designs preferred; a pull design is only allowed if G0 clears it; host-staged is known not to win (OPTLOG 128/189). If no pair performs, (a) is cancelled and effort moves to (e1)-(e3) |
| GPU0's root port differs from GPU1/GPU2's (`00:02.0` vs `00:03.x`) | Per-pair numbers in G0; the best pair is used for any 2-GPU design |
| Upstream merge conflicts (the fork merges upstream; `ggml-cuda.cu` and `ggml-backend-meta.cpp` are hot files, and upstream touched the meta backend on 09-23, 10-01 and 10-02) | New code in new files, small hooks, env toggles; rebase Phase 3/4 branches on each Kmic-68 tag; nothing pushed without your say |
| Correctness regressions and races (the fork shipped two cross-stream races that passed `test-backend-ops` for weeks) | Full gate, KLD, byte-identical checks where order is unchanged, race injection, long-context divergence test |
| Thermal limits (125 W cap; hot cards read up to 20% low) | Cool-card protocol, interleaved A/B, temperature logging, flag >72 °C |
| NCCL on Pascal (2.27.3 in the image; arch support unverified) and its BF16 prefill path | Verify NCCL init in Phase 2; patch prefill to f32/f16-exact before any accuracy claim |
| Toolchain: CUDA 13 dropped Pascal | Stay on CUDA 12.9 (current image) |
| `nsys` 2026.3 may not trace Pascal | Use `nvprof` in the container (the fork's scripts already do) |
| Issue #29466 (second-request crash under `-sm tensor`) | Second-request test in every gate; changes must not touch graph-reuse code without a reason |
| Model download in progress in the model directory | Nothing reads or writes there until you confirm it's complete; the comparison quant goes to a separate directory |

## 8. Server: cancelling a request when the client disconnects during prefill (added 2026-10-05; research only, no code change, nothing run)

Why this section exists: Phases 1.8 and 1.9 measured that a 64,000-token request dropped by its client halfway through
prefill kept the slot busy until prefill finished (about 84 s), at every `-b`. This section reads the code to find out
why. File and line numbers are the fork at `ae35056eb`. The fork has not changed `server-queue.cpp`, `server-queue.h`,
`server-http.cpp` or `server-stream.cpp` against its upstream merge-base (`f46bc30cb`, 2026-09-23); it has changed
`server-context.cpp` (+153 −9), but none of the changed lines touch cancellation, `should_stop` or the yield mechanism.

### 8.1 How a disconnect becomes a cancel, as written

1. **HTTP side.** `server-http.cpp:719` passes httplib's `req.is_connection_closed` to the handler as `req.should_stop`.
   httplib implements it as "socket not alive" (`vendor/cpp-httplib/httplib.cpp:1805`, a zero-timeout readability
   check and a 1-byte peek).
2. **Waiting for results.** The completion handler waits in `server_response_reader::next(should_stop)`
   (`server-queue.cpp:550`): before the first result at `server-context.cpp:4528` (stream) or `:4496` (non-stream),
   and after it inside the chunked provider at `:4609`. `next()` calls `recv_with_timeout(ids, 1 s)` and tests
   `should_stop()` **only when that call returns on timeout** (`HTTP_POLLING_SECONDS = 1`, `server-context.cpp:46`).
3. **Posting the cancel.** When `next()` returns null the response object is destroyed and
   `server_response_reader::stop()` (`server-queue.cpp:603`) logs `cancel task, id_task = N` and posts a
   `SERVER_TASK_TYPE_CANCEL` task to the **front** of the task queue.
4. **Handling the cancel.** `process_single_task` (`server-context.cpp:2504`) releases the slot that holds the task.
5. **The slot loop.** `update_slots()` (`:2872`) builds one batch in `pre_decode()` (prompt tokens are added at
   `:3623`, at most `n_batch` per pass), then calls `decode()` (`:3746`), which runs `llama_decode` inside
   `queue_tasks.yield_to_queue()` (`:3782`). While a decode is running, a worker thread serves the queue, but
   `process_single_task` **declines every task except METRICS and SLOT_GET** (`:2427`). A declined cancel is set aside
   and put back at the front of the queue when the decode returns (`server-queue.cpp:252`), so it is handled before
   the next batch is built.

### 8.2 Answers

**(1) Is cancellation checked between micro-batches during prompt processing?**

- Not inside `llama_decode`. A cancel is honoured only between `llama_decode` calls. There is no per-micro-batch check
  in `llama_context::decode`'s loop, and the CUDA and meta backends do not implement the abort callback
  (`llama_set_abort_callback` reaches only backends that export `ggml_backend_set_abort_callback`; `GGML_STATUS_ABORTED`
  is mapped to return code 2 at `src/llama-context.cpp:1588` and `:1963`).
- **Corrected 2026-10-05 after Test A.** An earlier version of this section said each `llama_decode` call during
  prefill carries about one micro-batch whatever `-b` is. That was read from `-b 2048` logs and is wrong at
  `-b 32768`. `pre_decode()` adds up to `n_batch` prompt tokens per pass, stopping early `4 + n_ubatch` and `4` tokens
  before the end of the prompt for checkpoints (`:3660`). A 64,000-token prompt at `-b 32768` is therefore four
  calls: 32,768, 29,180, 2,048 and 4 tokens. A cancel waits for the call in flight, up to about 75 s here.
- **Test A (`p100-docs/3gpu/RESULTS.md` sections 1.18 and 1.19) measured both delays on the serving configuration:** with no polling the
  cancel is logged 0.01 s after the drop and the slot is released 72–74 s later, at the end of the 29,180-token
  call; with `/slots` polled 5 times a second the cancel is not logged until 80 s after the drop, after prefill has
  finished. So the slot loop's batch granularity and the starved disconnect poll are both real.
- **The starved disconnect poll (confirmed by Test A, scenario 2).** `recv_with_timeout`
  (`server-queue.cpp:450`) waits on a condition variable for 1 s and returns null only on timeout. Every result sent
  for any waited task calls `notify_all` (`server-queue.cpp:489`); a woken waiter that finds nothing for itself
  starts a fresh 1 s wait. If results arrive more often than once a second, the timeout never fires and
  `should_stop()` is never called. The Phase 1.8 and 1.9 clients polled `/slots` five times a second after the
  drop, and since upstream PR 27041 those requests are answered during a decode. That is enough to starve the poll.
- **Consequence for our results:** the Phase 1.8 and 1.9 cancel figures (about 84 s at every `-b`) were taken with
  `/slots` polling and show only the starvation. Without polling, the wait is the decode call in flight, which `-b`
  sets; a smaller `-b` should cut it to about one batch, and that is not yet measured. The "second client waits
  for the whole prefill" result is unaffected: with one slot that is queueing, not cancellation.
- **The starvation is still a real defect**, independent of my test: a dashboard polling `/slots` or `/metrics` faster
  than 1 Hz, or another slot streaming tokens (each token is a result), would also keep a dropped request alive until
  its first token.
- **Where a check would go:**
  - *Fix A (the starvation):* in `server_response::recv_with_timeout`, wait against a fixed deadline (`wait_until`)
    so the function returns null one polling interval after it was called, however many unrelated notifications
    arrive. One function in `server-queue.cpp`.
  - *Fix B (only if a single `llama_decode` call can ever span many micro-batches, e.g. MTP off):* an atomic
    "cancel requested" flag per slot, set when the cancel task is declined during a yield; a check of it in
    `llama_context::decode`'s micro-batch loop (through the existing abort callback, called once per micro-batch
    by llama itself so it does not depend on backend support), returning 2; and handling of that return value in the
    server's `decode()`.

**(2) Does upstream master handle this?** Upstream master (`e117148a4`, 2026-10-05) has the same code as the fork in
every place above: `recv_with_timeout` is byte-identical, the first-result wait still uses `req.should_stop`, and
the decline-while-decoding rule is the same. So it has the same batch-granularity cancel and the same starvation.

| # | Kind | Title | State | Dates |
|---|---|---|---|---|
| [9679](https://github.com/ggml-org/llama.cpp/pull/9679) | PR | `server`: cancel prompt processing & non-streamed requests when connection closed | closed, superseded by 11285 | opened 2024-09-29 |
| [11285](https://github.com/ggml-org/llama.cpp/pull/11285) | PR | server : implement cancellable request | **merged** | 2025-01-18 |
| [24496](https://github.com/ggml-org/llama.cpp/issues/24496) | issue | server: generation not cancelled when client disconnects (is_connection_closed never checked) | closed, not planned | 2026-06-11 to 2026-06-15 |
| [24630](https://github.com/ggml-org/llama.cpp/pull/24630) | PR | server: cancel generation when client disconnects | closed, not merged | opened 2026-06-14 |
| [27041](https://github.com/ggml-org/llama.cpp/pull/27041) | PR | server: allow accessing /metrics and /slots during llama_decode() | **merged** | 2026-08-14 |
| [27481](https://github.com/ggml-org/llama.cpp/issues/27481) | issue | server: DELETE resumable stream can abort child or fail to cancel before first token | closed | opened 2026-08-21 |
| [27482](https://github.com/ggml-org/llama.cpp/pull/27482) | PR | server: cancel resumable streams before first token | **open** | opened 2026-08-21 |
| [29163](https://github.com/ggml-org/llama.cpp/issues/29163) | issue | Misc. bug: interrupted connection trashes KV cache | **open** | opened 2026-09-19 |

- 11285 is the mechanism described in 8.1. 27041 introduced the yield and the "decline everything but METRICS and
  SLOT_GET" rule.
- 27482 covers a different case (requests carrying `X-Conversation-Id`, which deliberately survive a socket close)
  but touches the same wait at `server-context.cpp:4528`; it would conflict textually with Fix A's neighbourhood only
  lightly.
- 29163 is the risk on the other side: after an interrupted request the next one reprocessed the whole
  conversation. Any change here must be tested for that (8.4, test C).
- I found no upstream issue or PR describing the starvation of the disconnect poll. Searches: "cancel", "disconnect"
  and "abort" with "prompt processing", plus title searches for client disconnect; GitHub search, 2026-10-05.

**(3) Size of the change and what could break.**

| | Files | Lines (estimate) | What could break |
|---|---|---|---|
| Fix A | `tools/server/server-queue.cpp` | about 10 | SSE ping timing (the same wait drives the ping interval at `server-context.cpp:4609`); requests that currently live through a brief disconnect check now end within about 1 s, as intended |
| Fix B | `tools/server/server-context.cpp`, `src/llama-context.cpp`, possibly `tools/server/server-queue.cpp` | about 60–90 | see below |

Fix A changes no slot, cache or model state: it only makes an existing check run on time. What a cancel does to the
slot is already exercised today whenever a client drops during generation.

Fix B interrupts a batch that is partly computed, which is new:

- **Slot state and KV cache.** `pre_decode()` appends the batch's tokens to `slot.prompt.tokens` before the decode
  (`:3646`). After an abort the KV cache holds only the micro-batches that completed, so the slot's token list and
  the cache disagree. The handler would have to truncate both to the last completed position (`seq_rm` plus a
  matching truncation of `slot.prompt.tokens`), or the next request's prefix match reuses positions that were never
  computed.
- **Hybrid/recurrent state.** This model has delta-net (recurrent) layers. Their state cannot be rolled back by
  position the way attention KV can; the server relies on checkpoints for that (`create_checkpoint`, `:2352`; restore
  at `:3468`). An abort mid-batch leaves recurrent state at an arbitrary position with no checkpoint there, so the
  safe handling is to drop back to the last checkpoint, which costs reprocessing on the next request (the symptom
  of issue 29163).
- **MTP draft context.** Under MTP every prompt position is mirrored into the draft context
  (`:3639`, "the streaming hook can mirror t_h_nextn into ctx_dft"), and checkpoints stash the draft's speculative
  state (`:2415`, `load_dft` at `:3469`). Target and draft would have to be truncated to the same position.
- **Async decode and timing.** Prompt batches without output are not synchronized (`:3791`); the fork's prompt
  timing assumes a later sync (`metrics_queue_prompt`, `:4195`). An abort path has to synchronize before touching
  state.
- **Return-code handling.** The server's `decode()` treats a non-zero return as an error or a KV-space retry
  (`:3797` onward). Return code 2 needs its own branch, or a cancel would be reported as a failure.
- **NCCL.** An abort between micro-batches leaves no exchange in flight; an abort inside one must not be attempted.

Recommendation after Test A: no code is needed for the batch-granularity delay; a smaller `-b` addresses it by
configuration (Phase 1.9 measured under 1% prefill cost down to `-b 2048`), once confirmed by a no-polling run at
that `-b`. Fix A is still the right change for the starvation. Fix B stays not worth its state risk.

**(4) How to test.** None of this has been run.

- **Test A, is there a problem at all (no code change; needs the test machine with serving stopped).** Start a
  64,000-token request, drop the connection at half the prefill time, and do **not** poll anything. Read the server
  log afterwards: time from the drop to `cancel task`, and from there to `stop processing`. Expected if the
  starvation theory is right: `cancel task` within about 1 s and the slot released within one micro-batch (about
  5–6 s at 2,048 tokens and 390 t/s). Repeat with `/slots` polled at 5 Hz from a second connection: expected to
  reproduce the 84 s. Three repetitions of each. This one test decides whether anything needs fixing.
- **Test B, slot frees within one micro-batch (after Fix A).** Same as test A with the 5 Hz poll running, plus a
  second variant on `-np 2` where the other slot is streaming a long generation during the drop. Pass: in every
  repetition the time from the drop to `stop processing` is under one micro-batch time plus 2 s, and a new request
  sent at the moment of the drop gets its first token within that time plus its own prefill.
- **Test C, the next request is unchanged.** With a fixed seed and greedy sampling (`--temp 0`), MTP on:
  1. reference: request R (a 20,000-token prompt, 200 tokens generated) on a fresh server; save the text and the
     prompt-token count it reports as processed.
  2. cancelled-then-R: on a fresh server send a different 64,000-token request, drop it mid-prefill, then send R.
     Pass: identical text to the reference.
  3. cancelled-then-continue: send a 64,000-token request X, drop it mid-prefill, send X again in full. Pass:
     identical text to X run uncancelled, and the log shows how many tokens were reused (this is where issue 29163
     would show up as a full reprocess, and where a wrong truncation would show up as different text).
  4. repeat 2 and 3 with the drop during generation, as a control for today's behaviour.
  Also run the fork's existing server tests that cover this area (`tools/server/tests/unit/test_completion.py`
  `test_cancel_request`, and `test_stream.py`).
- **Regression checks for Fix A:** the 90-minute soak of Phase 1.8 (no failed requests, no drift), and a streaming
  request left idle long enough to see SSE pings still arrive.

### 8.3 Open points

- Test A2 (2026-10-05) measured cancel latency with no polling at `-b 2048` and `-b 4096`: 3.8–4.3 s at both. The
  worst case (one full step) is still inferred. With `/slots` polled 5 times a second it stays about 80 s.
- Test A, the first test in 8.2 (4), has been run; tests B and C have not.
