#!/usr/bin/env bash
# One-minute in-model check: rebuild libggml-cuda if needed, then time one MTP verify pass
# (a 5-token batch through the full model, 2k context) with llama-bench.
#   tools/quick.sh                      # current build
#   tools/quick.sh A=GGML_X=0 B=GGML_X=1  # A/B two env settings, ABBA order, same binary
# The model sits in the page cache, so a run takes ~10 s. Single runs spread ~4%, so treat a
# difference under ~5% as noise and confirm anything you keep with depth-bench (ABBA, seeds).
set -euo pipefail
cd "$(dirname "$0")/.."
cmake --build build-opt --target ggml-cuda llama-bench -j 14 2>&1 | grep -E 'error|warning: .*mmvq' || true
# prints the median and min of 15 passes; the first pass after load is dropped
run() { env GGML_CUDA_P2P=1 "$@" ./build-opt/bin/llama-bench -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
          -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 5 -n 0 -ub 5 -b 5 -d 2048 -r 16 -o jsonl 2>/dev/null \
        | python3 -c 'import json,sys,statistics as S
for l in sys.stdin:
    t=sorted(x/1e6 for x in json.loads(l)["samples_ns"][1:]); print("median %.2f  min %.2f ms/pass" % (S.median(t), t[0]))'; }
if [ $# -eq 0 ]; then run; exit; fi
A="${1#A=}"; B="${2#B=}"
for arm in A B B A; do v=$([ $arm = A ] && echo "$A" || echo "$B"); printf '%s %-28s ' $arm "$v"; run $v; done
