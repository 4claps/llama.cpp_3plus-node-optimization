# Limitations

Read this before quoting a number from this directory.

## Scope

- **One machine.** Every number comes from a single host: three P100s on PCIe 3.0 x8 behind one host bridge, a
  Sandy Bridge-E CPU, 16 GB of RAM, a 125 W power cap. Nothing here was reproduced on a second system. The host-bound
  decode result in particular depends on a 2011 CPU.
- **One model family and three quants** (Qwen3.8-27B: unsloth UD-Q6_K_XL, unsloth Q8_0, and a local pure Q6_K).
- **One build** (`ae35056eb`), run in a container. llama-bench reports `build_commit=unknown` inside the container;
  the checkout was verified by hand.

## Open questions

- **2 GPUs against 3 GPUs is unresolved.** The only 2-GPU runs were in round 1, with CUDA graphs off and mmap
  loading. Both settings distort decode on this host, and the 2-GPU arms were not repeated. Round 1 showed 2 GPUs ahead
  of 3; that result should not be relied on in either direction. No 2-GPU run was made with NCCL.
- **Serving context size is not covered.** Server runs used `-c 4096` to `-c 65536`. The fork's documented
  `-c 262144` was not run on three cards with these files, and VRAM fit at that size is not measured.
- **No agent-level or end-to-end results.** There is no multi-turn, tool-calling or mixed prefill/decode workload
  here; only llama-bench, llama-perplexity, llama-speculative-simple and a few scripted server requests.

## NCCL

- **KLD was measured on one corpus at `-c 4096` only** (the fork's gate corpus, 32,768 tokens, q4_0 KV). Whether the
  difference grows with position at long context (KV cache and delta-net state accumulation) is **not measured**. At
  16,384 tokens only repeatability was checked.
- **NCCL is not bit-exact for decode.** It is repeatable (identical hashes across runs) but differs from the
  non-NCCL path: 3.3% of saved-logit bytes differ, and greedy generations diverge. Any check that compares bytes or
  greedy text against a non-NCCL reference fails under NCCL.
- **Prefill exchanges of 26 or more tokens use BF16** in the NCCL path as written (131,072 elements at 3 GPUs,
  5,120 per token). Decode, MTP draft and MTP verify exchanges stay in f32.
- Why a different f32 summation order grows to a mean KLD of 0.002 was not investigated. The comparison offered
  (`-ub 5` against `-ub 2048` on the non-NCCL build gives 0.0023) shows the size is not specific to NCCL; it does
  not explain it.
- NCCL chooses a host-memory transport by default on this topology; the results use `NCCL_P2P_LEVEL=SYS`. The
  host-memory path was measured only for speed, and its behaviour with a small `/dev/shm` was not tested.

## MTP

- **MTP settings were compared with greedy decoding** (`--temp 0 --top-k 1`), 256 tokens, three fixed prompts. The
  fork's serving configuration samples (`--temp 1.0` with `LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20`).
  The ranking of p-min values may not hold under sampling. The fork's own default (n-max 4, p-min 0.2) was not in the
  grid.
- Under greedy decoding, repetitions measure timing noise only: acceptance is fixed per configuration and prompt.
- Differences between the NCCL and non-NCCL MTP results come partly from different generated text, not only speed.

## Attribution and profiling

- **The kernel table is approximate.** Kernel times come from nvprof with graphs on; shapes come from the fork's op
  profiler, which only records with graphs off. Per-shape times were scaled per quant type to the nvprof totals
  (Q8_0 ×0.959, Q6_K ×1.034). The op profiler prints only its top 45 entries per device, so Q5_K is under-listed
  (0.51 of 1.36 ms) and 4% of matvec time is not attributed to a shape. "Lost to the ceiling" uses the fork's
  605 GB/s figure, not a ceiling measured here.
- **nvprof inflates host-side time** (token time ×1.4 with graphs on, ×2.1 with graphs off). Kernel durations were
  used as measured; idle time was rescaled to the unprofiled token time and split in the trace's proportions. The
  split between "exchange-adjacent" and "other" idle is therefore an estimate.
- **Round 1's decode exchange share (19%) is superseded.** With graphs off the idle it counted was mostly the GPUs
  waiting for the host. With graphs on the same definition gives 11.3%.
- **The AllReduce shootout is a standalone program**, not the fork's exchange. Its (a) arm replicates the generic
  pattern with peer copies and add kernels; it matches the in-model prefill latency (32.7 against 31.0 ms) but is not
  the same code. Savings per token or per step are projections from measured exchange counts.

## Protocol deviations

- **Cool-down** was "all GPUs ≤45 °C or ≤ idle baseline + 2 °C" in round 1, "≤45 °C" in round 2, and "≤48 °C" with
  the fans at full speed for most of round 3. Start temperatures are recorded per run. The first three round-3
  results were taken under the ≤45 °C rule.
- **nvprof and op-profile runs in round 1 used `--no-warmup`.**
- **llama-bench has no `-c`**; its context is prompt + generation + depth.
- **Round 2 and round 3 loaded the model with `-lm none`; round 1 used the default (mmap).**
- Several runs were skipped or killed (out of memory on 2 GPUs with Q8_0 at depth 65536; a watchdog stop on a
  power-capped clock dip under layer split). RESULTS.md section 3.8 lists them.

## Other

- **The pure Q6_K is a local requantization of the unsloth Q8_0** (`--allow-requantize --pure`, no imatrix). It
  matches the fork's described reference in size and tensor types, but it is not the fork's file, and requantizing
  from Q8_0 is not the same as quantizing from full precision. Its quality was not measured.
- **Upstream issue 29466** (second-request assert under `-sm tensor`) did not reproduce in four scripted server runs.
  That is not evidence that it is fixed.
- **Raw data is incomplete for rounds 2 and 3**: see [results/README.md](results/README.md).
- Times in logs are the test host's local time (EDT).
