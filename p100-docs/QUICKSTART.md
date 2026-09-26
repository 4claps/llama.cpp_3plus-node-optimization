# Quickstart

The binaries are built for sm_60 only.

## Put it on PATH

    export PATH="/mnt/fast/p100-llamacpp-release/bin:$PATH"

`bin/` holds a small wrapper for every program in `build/`, plus `qwen-server`. Use the
wrappers, not `build/` directly. Each wrapper sets `LD_LIBRARY_PATH` to the bundle. The real
binaries carry a RUNPATH back to the tree they were compiled in, so run from `build/` they can
quietly load a different build's `libggml-cuda.so`. To check which one is live:

    LD_DEBUG=libs llama-cli --version 2>&1 | grep -m1 "trying file=.*ggml-cuda"

## Serving

`qwen-server` runs the configuration below. Extra arguments are appended and override the
defaults, so `qwen-server --port 9000` works, and `QWEN_MODEL=/path/to.gguf qwen-server` swaps
the model.

**Text only:**

    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3 \
    LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
    llama-server \
      -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 32768 -ub 2048 -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
      -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
      --jinja \
      --host 0.0.0.0 --port 8080 \
      --tools all \
      --mcp-servers-config ~/mcp-servers.json

**With vision:** add the projector and halve the ubatch.

      --mmproj /mnt/fast/models/mmproj-Qwen3.8-27B-Q8_0.gguf
      -ub 1024                     # instead of 2048

`qwen-server` also passes the model card's sampling, `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0`. The gguf carries the first three itself, but not min-p, so without the flag llama.cpp's 0.05 applies. Don't change these: other values make the model loop.

### What the flags do

| flag | why |
|---|---|
| `-sm tensor` | splits every layer across both cards. The model doesn't fit at 262144 context otherwise |
| `-fa 1` | flash attention. Tensor split requires it |
| `-ctk q4_0 -ctv q4_0` | q4_0 KV cache. An f16 cache doesn't fit at this context |
| `-c 262144` | the model's full context. Allocating it costs nothing on decode. Only filling it does |
| `-np 1` | one server slot. Each slot allocates its own full KV cache, and without this the server sizes several and fails at startup |
| `-b 32768` | the most tokens one decode call may take. **Needed for MTP at long context:** with `-b 262144` the whole prompt becomes one batch, and on a 259k-token prompt draft acceptance falls to 0 and decode to 5.4 t/s. At 32768 the same prompt gives 0.98 acceptance and 26.1 t/s |
| `-ub 2048` | tokens per GPU pass. It sets prefill speed and VRAM use; see below |
| `--spec-type draft-mtp` | speculative decoding with the model's built-in MTP head. MTP isn't a separate model: the `*-MTP-ONLY` gguf doesn't load on its own |
| `--spec-draft-n-max 4 --spec-draft-p-min 0.2` | draft up to 4 tokens, and stop drafting below 20% confidence. See below for 3 vs 4 |
| `-ngld 99` | the draft layer on the GPU |
| `-ubd 64` | the draft context's own ubatch. Without it the draft inherits `-ub` and reserves a second copy of the attention mask, which runs out of memory at full context. 64 is also faster than 256 (23.0 against 21.4 t/s) |
| `-ctkd q4_0 -ctvd q4_0` | q4_0 for the *draft's* KV cache, which is otherwise f16. 151 MB instead of 537, at no measurable cost to acceptance |
| `GGML_CUDA_P2P=1` | direct copies between the cards. Without it they go through host memory |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=3` | CUDA graphs for single-token graphs only (the MTP draft steps). Full graphs (`1`) run VRAM out at full context when instantiated for the big verify and prefill graphs; `0` turns them off entirely (OPTLOG 212) |
| `LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20` | the MTP draft samples from its own top 20 at temperature 1.0, and the verify uses the speculative-sampling rule (accept with min(1, p/q), else draw from the residual). The output distribution is exactly the model card's sampling; only acceptance changes: +15% tokens per cycle at 2k, +8% at 260k (OPTLOG 202) |
| `--jinja` | use the chat template stored in the gguf. Tool calls need it |
| `--tools all`, `--mcp-servers-config` | the server's built-in tools, plus MCP servers from that file |
| `--host 0.0.0.0 --port 8080` | listen on the LAN |

### VRAM

All figures are **per card**. GPU0 is the one to watch: it also carries the desktop (Sunshine
holds ~392 MiB there), and the vision projector loads onto it whole.

VRAM use grows as the context fills. The attention mask is sized to the *used* part of the cache
times the ubatch, so a short prompt says nothing about a full one. At full depth these
configurations bottom out at:

| configuration | GPU0 free at the low point of a 259k-token prefill |
|---|---|
| text, `-ub 2048` | 757 MiB |
| vision, `-ub 1024` | 731 MiB |

`-ub` is the lever. Each unit costs ~0.74 MiB on GPU0, and lowering it costs only prefill speed:

| text `-ub` | prefill at full depth | GPU0 low point |
|---|---|---|
| 2048 | 137.4 t/s | 757 MiB |
| 256 | 119.5 t/s | 2205 MiB |

If anything else shares GPU0, like a browser or a second display client, drop one step: text to
`-ub 1024`, vision to `-ub 512`. Vision at `-ub 2048` fails during load.

### Draft length: 3 or 4

With `--spec-draft-p-min 0.2`, drafting almost never stops early (3.98 of 4 drafts on average),
so every verify is `n_max + 1` tokens wide. A wider verify is cheap at short context and costly at
depth, where attention runs once per verify token. Interleaved on one build, real sampling:

| context | n_max 3 against 4 |
|---|---|
| 2k | −3% t/s |
| 64k | −8% |
| 260k | **+17%** (24.1 against 20.6 t/s) |

So 4 (the default in `qwen-server`) for most work, and 3 if you live past ~150k. Single runs
spread by ~10%, because acceptance depends on the text, so compare several.

MTP through the server, 2026-09-23 build, `--temp 0.3 --top-k 20`: 38-50 t/s at 2k, 36-38 at
64k, 24-27 at 260k.

## Benchmarking decode

    GGML_CUDA_P2P=1 llama-bench -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
      -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5

Leave `GGML_CUDA_GRAPHS_PRE_VOLTA` unset here. At long context, time at least 512 generated
tokens: `-n 128` is dominated by a 2-3 s first-token cost and reads far too low.

## Precision switches

| variable | effect |
|---|---|
| `GGML_CUDA_GEMM_FOLD=0` | prefill matmuls back on cuBLAS fp16 (whole-K fp16 accumulation). The default since 2026-09-25 is the fold kernel: fp16 products, fp32 accumulation, ~6% slower at short context and 4x more accurate per op; KLD against an all-fp32 run 0.00152 → 0.00125, where fp32 with a different summation order alone gives 0.0006-0.001 (OPTLOG 221). `=1` keeps fp32 outputs (no more accurate, and it turns off the f16 peer exchange) |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | prefill matmuls fully in fp32 cuBLAS. Exact, and ~40% slower prefill. Since the fold kernel it no longer buys a measurable difference in perplexity |
| `GGML_CUDA_FA_GEMM=0` | turns off the cuBLAS-GEMM attention path used for long prefill (on by default at batch ≥ 128 and KV ≥ 4096). The tile kernel is more accurate per op but slower at depth. Perplexity can't tell them apart |
| `GGML_CUDA_FA_GEMM_PREC=32` | fp32 accumulation inside the GEMM attention path. Slower (-11% to -27% prefill) and buys nothing measurable |
| `GGML_CUDA_FA_TILE_Q4_0=0` | turns off direct q4_0 dequant in the tile kernel. For A/B testing only |

## Checking a build

    /mnt/fast/p100-llamacpp-release/tools/gate.sh

It runs the decode benchmark, then perplexity on the right corpus, then the flash-attention op
tests. Expect perplexity near 2.6101 ± 0.0198. The gate band is 2.6209 ± 0.0199. Use the script rather
than typing the command: a different text file reads ~2.7566 on *any* build, which looks like a
regression and isn't.
