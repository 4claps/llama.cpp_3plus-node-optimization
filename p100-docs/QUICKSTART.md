# Quickstart

For two Tesla P100-16GB cards (or other Pascal sm_60 cards) running Qwen3.8-27B Q6_K with its
built-in MTP head. Build first: see [BUILD.md](BUILD.md).

## Run the server

**Text only:**

    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3 \
    LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
    ./build-opt/bin/llama-server \
      -m /path/to/Qwen3.8-27B-Q6_K.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 32768 -ub 2048 -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
      -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
      --jinja --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
      --host 0.0.0.0 --port 8080

**With vision:** the same command, plus the projector, with the ubatch lowered to 1024:

      --mmproj /path/to/mmproj-Qwen3.8-27B-Q8_0.gguf -ub 1024

## What to expect

Prefill, and decode with MTP, at each context depth:

| depth | prefill | decode |
|---|---|---|
| 2k | 316 t/s | 52 t/s |
| 32k | 342 t/s | 56 t/s |
| 62k | 257 t/s | 46 t/s |
| 92k | 194 t/s | 38 t/s |
| 122k | 162 t/s | 35 t/s |
| 152k | 141 t/s | 30 t/s |
| 182k | 125 t/s | 31 t/s |
| 212k | 118 t/s | 33 t/s |
| 242k | 108 t/s | 28 t/s |
| 260k | 100 t/s | 28-34 t/s |

MTP decode depends on how predictable the text is: code and factual answers run faster than
creative writing. Cards that have been under sustained load read ~5-10% lower.

## What the flags do

| flag | why |
|---|---|
| `-sm tensor` | splits every layer across both cards. Needed to fit the full context |
| `-fa 1` | flash attention. Tensor split requires it |
| `-ctk q4_0 -ctv q4_0` | q4_0 KV cache. An f16 cache doesn't fit at 262k |
| `-c 262144` | the model's full context. Reserving it costs nothing until it fills |
| `-np 1` | one server slot. Each slot allocates its own full KV cache |
| `-b 32768` | **needed for MTP at long context.** A larger batch turns a long prompt into one huge batch, and draft acceptance collapses |
| `-ub 2048` / `-ub 1024` | tokens per GPU pass. 2048 for text (faster prefill); 1024 with vision, which needs the VRAM headroom |
| `--spec-type draft-mtp` | speculative decoding with the model's built-in MTP head |
| `--spec-draft-n-max 4 --spec-draft-p-min 0.2` | draft up to 4 tokens; stop below 20% confidence. Use 3 if you mostly work past ~150k context |
| `-ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0` | the draft layer on the GPU, with its own small ubatch and a q4_0 cache. Without `-ubd 64` the draft runs out of memory at full context |
| `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0` | the model card's sampling. The gguf doesn't carry min-p, so without the flag llama.cpp's 0.05 applies |
| `GGML_CUDA_P2P=1` | direct copies between the cards instead of through host memory |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=3` | CUDA graphs for the single-token MTP draft steps only. Full graphs (`1`) run out of VRAM at full context |
| `LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20` | the draft samples instead of taking its top token, and the verify uses the speculative-sampling rule. The output distribution is unchanged; more drafts get accepted |

## VRAM

Figures are **per card**, at a full 262k context, with nothing else on the GPU. GPU0 is the
tighter card because the vision projector loads onto it.

| configuration | GPU0 free at full context |
|---|---|
| text only, `-ub 2048` | ~1.1 GiB (estimated) |
| vision, `-ub 1024` | ~1 GiB |
| vision, `-ub 2048` | ~0.5 GiB: too tight |

VRAM use grows as the context fills, so check it with a full prompt, not a short one. If GPU0
also drives a display or runs other programs, subtract what they use. Lowering `-ub` is the
fix, and it costs only prefill speed: 2048 → 1024 → 512.

## Precision switches

Everything defaults to the fast path, which is at least as accurate as stock. These exist for
A/B testing.

| variable | effect |
|---|---|
| `GGML_CUDA_GEMM_FOLD=0` | prefill matmuls on stock cuBLAS fp16 instead of the fold kernel (fp16 products, fp32 sums) |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | prefill matmuls fully in fp32. ~40% slower, no measurable accuracy gain |
| `GGML_CUDA_FA_GEMM=0` | turns off the GEMM attention path for long prefill |
| `LLAMA_MTP_DRAFT_VOCAB=0` | the MTP draft scores the full vocabulary instead of a small copy of the common tokens. Saves ~112 MiB per card, drafts get slower, output is unchanged |

## Benchmarking

    GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m /path/to/Qwen3.8-27B-Q6_K.gguf \
      -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5

Expect ~31 t/s (plain decode, no MTP). Measure on cool cards: right after a long run, P100s can
read up to 20% low. To check accuracy, run `tools/gate.sh`; perplexity should land near 2.61.
