#!/usr/bin/env bash
# exp.sh TAG ONLY(qk2|pv2) patch.py : apply patch to fattn-gemm.cu, build cubin(s) into st/TAG, restore, profile
# needs /mnt/fast/p100-scratch/fa-prof2.sh (faacc + nsys) and st/cur holding the baseline cubins (FACUBIN_DIR=.../st/cur facubin.py)
set -e
T=$1; ONLY=$2; PATCH=$3; S=/mnt/fast/p100-scratch/st; L=$HOME/llama-opt; F=$L/ggml/src/ggml-cuda/fattn-gemm.cu
cp $F $S/fattn-gemm.cu.keep
python3 $PATCH $F || { cp $S/fattn-gemm.cu.keep $F; exit 1; }
FACUBIN_ONLY="$ONLY" FACUBIN_DIR=$S/$T python3 $L/p100-handoff/tools/sass-gemm/facubin.py 2>&1 | grep -E 'conflicts after|register count|rror' || true
cp $S/fattn-gemm.cu.keep $F
for k in qk2 pv2; do [ -f $S/$T/$k.cubin ] || cp $S/cur/$k.cubin $S/$T/; done
cd /mnt/fast/p100-scratch; SD=$S/$T REPS=3 timeout 60 ./fa-prof2.sh 2>&1 | grep -E 'ms/call'
