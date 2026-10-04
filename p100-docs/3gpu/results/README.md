# Raw results

About 4 MB. Three rounds, in the order they were run. RESULTS.md is built from these and from files that are **not**
included (listed at the end).

## round1/ (graphs off, mmap; complete raw output)

- `runs/<run>.out`, `runs/<run>.err`: stdout and stderr of every run (llama-bench CSV after the container banner;
  the P2P programs' CSV for `g0_*`).
- `telemetry/<run>.smi.csv`: the full 1 Hz `nvidia-smi` log of every run (small enough to include);
  `telemetry/fanlog.csv`: fan PWM and RPM for the last part of the round.
- `telemetry_summary.csv`: per run and GPU: peak temperature, mean utilization, median and minimum SM clock while
  busy, mean power while busy.
- `exit_codes.csv`, `commands.txt`, `run_log.txt` (cool-down waits, start temperatures, command lines),
  `skipped_runs.txt`.
- `nvprof-summaries/`: decode attribution for all configurations, prefill attribution for the five profiled prefill
  runs, and the op-profile output (per-op times and shapes) for four runs.

## round2/ and round3/ (per-run summaries only)

- `runs.csv`: one row per run: exit code, cool-down wait, start temperatures, test, mean and stdev t/s, and for MTP
  runs acceptance and draft counts. **Derived from the queue's summary lines, not from the raw output.**
- `run_log.txt`: every command line, cool-down wait and start temperature, and the queue's stop and resume events.
- round2: `decode_budget_graphs_off.txt` / `_on.txt` (analyzer output), `host_timeline_graphs_off.txt` / `_on.txt`
  (the server's `TL` phase marks, microseconds), `allreduce_shootout.csv`, `layer_split_overlap.txt`.
- round3: `kernel_profile_table.txt`, `op_profile_graphs_off_64_tokens.txt`, `server_second_request_c65536.jsonl`.

## round4/ (per-run summaries only)

- `runs.csv`: one row per run, as for rounds 2 and 3, with extra columns for the context-fit runs (peak memory per
  GPU, prompt tokens, prefill t/s). Derived from the queue's summary lines.
- `run_log.txt`: every command line, cool-down wait and start temperature. Host paths and the server address are
  replaced by `$ROOT`, `$P1`, `$MODELS2`, `$QUANT` and `$HOST_IP`.
- `kld_*_c16384.txt`: output of `scripts/kldpos.cpp` for the two pairs, with the per-position buckets.
- `battery_runs.csv`: one row per task run: configuration, server block, task, repetition, exit code, wall time,
  whether it was killed at the limit, whether it passed. `battery_report_A.txt` / `_B.txt`: the harness's summary.
- `battery_server_blocks.csv`: per server block: requests, prompt and generated tokens, prefill and decode t/s,
  draft tokens generated and accepted, peak memory per GPU.

## Left out

- **nvprof traces** (13–119 MB each, about 550 MB in total).
- **Raw per-run stdout, stderr and 1 Hz telemetry for rounds 2, 3 and 4**, including the battery's server logs. Only
  the summaries above are here.
- **The agent battery's harness, task definitions and agent transcripts** (a separate private project).
- **The round 4 queue and battery-runner scripts** (they hold machine-specific paths and addresses; their command
  lines are in `round4/run_log.txt`).
- **Saved logits** for the KLD runs (8–16 GB per file) and the model files.
- **Per-core CPU frequency logs** (rounds 2, 3 and 4).
