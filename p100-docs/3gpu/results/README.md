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

## Left out

- **nvprof traces** (13–119 MB each, about 550 MB in total).
- **Raw per-run stdout, stderr and 1 Hz telemetry for rounds 2 and 3.** They remain on the test machine and were
  not copied when this directory was assembled; only the summaries above are here.
- **Saved logits** for the KLD runs (8 GB per file) and the model files.
- **Per-core CPU frequency logs** (rounds 2 and 3).
