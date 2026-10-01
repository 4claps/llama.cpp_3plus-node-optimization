#!/usr/bin/env bash
# ~1-minute prefill A/B: one pp2048 -ub 2048 pass per arm (after llama-bench's warmup), A then B.
#   tools/ab-pp.sh "GGML_X=0" "GGML_X=1"
# Single passes spread ~1%: use it to screen; confirm keepers with a full ABBA (-r 2+) before committing.
# Prints GPU temps next to each result, since warm cards read low.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | paste -sd/)
  printf '%-40s %sC  ' "$1" "$t"
  env GGML_CUDA_P2P=1 $1 ./build-opt/bin/llama-bench -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf -sm tensor -fa 1 \
    -ctk q4_0 -ctv q4_0 -p 2048 -n 0 -ub 2048 -r 1 -o csv 2>/dev/null | tail -1 | awk -F, '{gsub(/"/,"",$(NF-1)); print $(NF-1) " t/s"}'; }
for a in "$@"; do run "$a"; done
