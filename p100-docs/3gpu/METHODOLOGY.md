# Methodology

## Build and image

- **Source:** this fork at `ae35056eb` (tag `p100-optimizations-b11515-ae35056`), unmodified.
- **Image:** a CUDA 12.9.1 development image (nvcc 12.9, NCCL 2.27.3 headers and libraries present). The build and
  every run happen inside it; the container gets all GPUs (`--gpus all`), the source tree and the model directory
  (read-only).
- **Build `build-opt`** (configuration "A"): `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release`,
  native CPU flags, `GGML_CUDA_NCCL=OFF`.
- **Build `build-nccl`** (configuration "B"): the same source and flags with `-DGGML_CUDA_NCCL=ON -DGGML_NATIVE=ON`, in
  a separate build directory. A first attempt with `-DGGML_NATIVE=OFF` crashed with an illegal instruction on this
  CPU (no AVX2) and was discarded.
- llama-bench prints `build_commit=unknown` in the container because git refuses the bind-mounted checkout; the
  checkout was verified at `ae35056eb` on the host.
- Host: NVIDIA driver 580.178.04, IOMMU off, performance cpufreq governor, all three cards behind one host bridge.

## Protocol

- **One run = one process in a fresh container.** The model is loaded from disk every time.
- **Cool-down before every run:** wait until every GPU is at or below the limit, with a 10-minute cap; the wait and
  the start temperatures are logged. The limit was ≤45 °C or idle baseline + 2 °C (round 1), ≤45 °C (round 2), and
  ≤48 °C with the fans at full speed (round 3 after its first three results).
- **Warmup:** llama-bench's built-in warmup (discarded) on every throughput run. Server runs send a discarded warmup
  request first. Round 1's nvprof and op-profile runs used `--no-warmup`.
- **Repetitions:** llama-bench `-r 3` per invocation unless stated; the tables give mean ± stdev of those
  repetitions. MTP cells in round 3 are three separate invocations.
- **Decode test:** `-p 0 -n 512` (tg512). **Prefill tests:** `-p 2048 -n 0`, with `-d 16384` or `-d 65536` for
  depth, and `-p 16384` / `-p 65536` for long prompts. llama-bench has no `-c`; its context is prompt + generation
  + depth.
- **Interleaving:** arms alternate per test (A B, then B A), or run in rotated order per round, so that drift does
  not favour one arm.
- **Telemetry:** `nvidia-smi` at 1 Hz for every run (temperature, SM clock, power, utilization, memory, clock
  event reasons), plus the cpufreq governor and per-core MHz at 1 Hz from round 2, and fan RPM and PWM every 2–5 s.
- **Watchdog:** a run is killed and the queue stops on GPU temperature above 80 °C, SM clock below 1000 MHz for 5 s
  while busy, no GPU memory in use for 10 minutes, a failed fan check, or a new kernel Xid. A new error in a run's
  stderr also stops the queue (round 1 only flagged it); an out-of-memory failure is recorded as "does not fit".
- **Nothing else runs on the machine** during a measured run (no builds, quantization or large copies).
- **Model load:** default (mmap) in round 1; `-lm none` in rounds 2 and 3. On this 16 GB-RAM machine mmap adds
  noise and a mean penalty to decode (RESULTS.md section 2.3) and roughly doubles load time.

## Command lines

Every command actually run is in `results/round1/commands.txt` and in the `RUN` lines of `results/round*/run_log.txt`
(`$BENCH_DIR` and `$SCRATCH` stand for local directories). The container is started as:

```
docker run --gpus all -e GGML_CUDA_P2P=1 [-e ...] \
    -v <llama.cpp checkout>:/workspace/src -v <models>:/models:ro -v <bench dir>:/phase1 \
    -w /workspace/src <image> <command>
```

Common arguments below: `COMMON = -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 -b 32768 -ub 2048`.

**Throughput (configuration A; B uses `build-nccl/bin` and adds `-e NCCL_P2P_LEVEL=SYS`):**

```
GGML_CUDA_GRAPHS_PRE_VOLTA=3 ./build-opt/bin/llama-bench -lm none -m XL.gguf COMMON -p 0 -n 512 -r 3 -o csv
GGML_CUDA_GRAPHS_PRE_VOLTA=3 ./build-opt/bin/llama-bench -lm none -m XL.gguf COMMON -p 2048 -n 0 [-d 16384|-d 65536] -r 3 -o csv
```

Round 1 ran the same commands without `GGML_CUDA_GRAPHS_PRE_VOLTA` and without `-lm none`; 2-GPU arms added
`CUDA_VISIBLE_DEVICES=1,2` or `0,1` and `-ts 1/1`; layer split replaced `-sm tensor` with `-sm layer`.

**MTP** (adapted from the fork's `p100-handoff/bench/mtp.sh`: same flags, with the model, a prompt file, `-n 256`,
`-c 16384`, `-b 32768 -ub 2048`, `-fit off`, `-lm none` and the graph variable added):

```
GGML_CUDA_GRAPHS_PRE_VOLTA=3 ./build-opt/bin/llama-speculative-simple -m XL.gguf --spec-type draft-mtp \
    --spec-draft-n-max N --spec-draft-p-min P -ngld 99 -f PROMPT -n 256 --temp 0 --top-k 1 --seed 42 \
    -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 -c 16384 -b 32768 -ub 2048 -fit off -lm none
```

The three prompts are raw chat-template text ending in an empty think block: a short chat question (about 40
tokens), a code-generation request (about 60 tokens), and a summarization request over the first 30,500 bytes of
`docs/build.md` (8,009 tokens). `-fit off` avoids the tool's auto-fit, which is not implemented for tensor split.

**Decode budget** (round 2) and **kernel profile** (round 3):

```
nvprof --print-gpu-trace [--print-api-trace] --csv --normalized-time-unit us --log-file trace.csv \
    ./build-opt/bin/llama-bench -lm none -m XL.gguf COMMON -p 0 -n 36 -r 1 -o csv
GGML_CUDA_OP_PROFILE=2 ./build-opt/bin/llama-bench -lm none -m XL.gguf COMMON -p 0 -n 64 -r 1 --no-warmup -o csv
LLAMA_TL=1 ./build-opt/bin/llama-server -m XL.gguf COMMON -np 1 -fit off -lm none -c 4096 ...   # host timeline
```

Analysis: `scripts/analyze_i1.py` (budget), `scripts/analyze_i4.py` (kernel table), `scripts/analyze_prefill.py`
and `scripts/analyze_decode.py` (round 1 attribution), `scripts/analyze_overlap.py` (layer-split overlap). The token
boundary in a trace is the device-0 logits copy to host; the last 32 of 36 tokens are analysed.

**NCCL quality** (round 3; corpus = the fork's gate corpus `p100-handoff/ppl-orig.txt`):

```
# baseline, non-NCCL build, at -ub 2048 and at -ub 5
./build-opt/bin/llama-perplexity -m XL.gguf -f ppl-orig.txt -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 \
    -c 4096 -b 4096 --chunks 8 -lm none -fit off -ub 2048 --kl-divergence-base base_ub2048.bin
# NCCL build against that baseline
NCCL_P2P_LEVEL=SYS build-nccl/bin/llama-perplexity <same arguments> -ub 2048 --kl-divergence --kl-divergence-base base_ub2048.bin
# repeatability: one 16,384-token chunk, saved logits hashed after each of 8 runs
NCCL_P2P_LEVEL=SYS build-nccl/bin/llama-perplexity ... -c 16384 -b 16384 -ub 2048 --chunks 1 --kl-divergence-base rep.bin
```

The server check sent seven chat requests to one server (a prompt, a different prompt, the first again, a follow-up
turn, a prompt of about 7,600 tokens, then a short one) with the default prompt cache, at `-c 32768` and `-c 65536`.

**P2P and the shootout:** `scripts/p2pbench/` (see `scripts/README.md`).

## The pure Q6_K file

The fork's documents describe their reference as a pure Q6_K of Qwen3.8-27B. No such file is published, so one was
made locally:

```
./build-opt/bin/llama-quantize --allow-requantize --pure Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-pure-Q6_K.gguf Q6_K
```

- Source: the unsloth pure Q8_0 (29,047,086,048 bytes), which carries the MTP block.
- Result: 22,431,001,568 bytes, 20.88 GiB of tensor data, all 506 quantized tensors Q6_K (including
  `output.weight`, `token_embd.weight`, `ssm_alpha` and `ssm_beta`), no Q8_0 tensors, MTP block present
  (`blk.64`, `nextn_predict_layers = 1`). 10.5 minutes of CPU time, with no GPU run in parallel.
- **This is not the fork's file.** It matches the description in size and tensor types, but it was requantized from
  Q8_0 rather than quantized from full precision, and no importance matrix was used. It was used for speed
  comparison only; its quality was not measured.
