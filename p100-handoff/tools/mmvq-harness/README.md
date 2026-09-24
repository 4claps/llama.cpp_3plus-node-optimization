# mmvq-harness: iterate on a q6_K multi-column matvec in seconds

    python3 extract.py            # once: real per-GPU weight slices from the gguf (~73 MB, run in a scratch dir)
    nvcc -O3 -arch=sm_60 -DCAND='"mykernel.cuh"' -o h harness.cu     # ~4 s
    ./h gate_8704x5120.q6k 8704 5120 5                               # ~3 s

A candidate defines `cand_scratch()` and `cand_run()` (see `ref_f32.cuh`). The harness prints
NMSE against a double-precision reference, next to a CPU model of today's q8_1 path on the same
data, and the GPU time (median of 200 runs). Activations are Gaussian with 0.2% outlier channels
at 60x (argument 5).

Today's kernel on the same shapes (test-backend-ops perf, CUDA0, 2026-09-24): 8704x5120 n=1 83 us,
n=5 154 us; 5120x8704 n=5 156 us. Isolated timings run at ~1328 MHz; the server runs at ~1189.
Then `tools/quick.sh` for the in-model check (~35 s).
