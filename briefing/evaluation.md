# Evaluation

## What a method submits

A method is a **scheduler program**.  Register it with a manifest
`outputs/schedulers/<name>.json`:

```json
{"command": ["python", "path/to/your_scheduler.py", "--any", "args"]}
```

The command is started with the workspace root as working directory
(`python`/`python3` resolve to the evaluator's interpreter).  Every manifest in
that directory is scored, by both evaluators, on every experiment.  The four
shipped names (`fairbatching`, `fairbatching_pab`, `vllm_vanilla`,
`vllm_sarathi`) are always scored; a missing or unreadable manifest scores as
a failed run.

## Protocol (JSON lines over stdin/stdout)

The evaluator owns the workload, the clock, the hardware and all latency
bookkeeping; the scheduler only makes decisions.  One run = one workload
instance = one fresh process.

1. `{"type": "init", "profile": {"a": .., "b": .., "c": .., "max_batch_tokens": 8192}}`
   → reply anything (e.g. `{"ok": true}`).  `profile` is an offline linear fit
   `step_ms ≈ a + b * new_tokens + c * total_context` of the hardware.  It is
   an approximation: true step times also depend on prefill attention cost and
   jitter.
2. Repeated `{"type": "step", "now": t_ms, "last_step_ms": d | null,
   "arrived": [rid, ...], "requests": [[rid, arrival_ms, prompt_len,
   prefill_done, generated, ttft_slo_ms, tpot_slo_ms], ...]}`
   → reply `{"reject": [rid, ...], "batch": [[rid, new_tokens], ...]}`.
   `requests` lists every admitted, unfinished request (including the ones in
   `arrived`, which are new this step); `last_step_ms` is the measured
   duration of the previous batch.
3. `{"type": "end"}` → exit.

Rules, checked at every step (any violation, a crash or malformed output
fails the run):

- `reject` may name only requests whose prefill has not started;
- a request appears at most once per batch; a prefill item gets
  `1 ≤ new_tokens ≤ prompt_len − prefill_done`; a decode item gets exactly 1;
- total `new_tokens` per batch ≤ `max_batch_tokens` (8192);
- an empty batch idles the engine for 5 ms (or until the next arrival);
- a scheduler must keep reading its input and answer each message within
  30 s; it may use at most 300 s of its own CPU time per run and 3600 s over
  all its runs of one evaluation (the reference uses ~1 s per run), and a run
  whose wall-clock time exceeds max(600 s, 100x its CPU time) counts as stalled.  These
  are validity limits measured on the scheduler process, not metrics:
  simulated time never depends on how long the scheduler thinks.

The batch runs from `now` to `now + duration`; all tokens it produces are
emitted at its end.  A request finishes when its (hidden) output length is
reached.  Requests still unfinished 600 s after the last arrival fail.
`data/sched_client.py` implements the loop; subclass `Policy` and call
`serve()`.  `data/serving_sim.py` is the same simulator the train evaluator
uses (hardware model, workload sampling, metrics) for local development.

## Workloads

Instances of 400 requests are sampled from `data/qwen_trace_head.csv` (first
2500 text requests of the Qwen-Bailian trace A): a window of real arrivals,
rescaled so the offered rate is exactly the target RPS, with prompt/output
length pairs bootstrapped from around that window and random request ids.
Within a load sweep each workload instance is replayed at every load (only
the time scale changes), as the paper replays its trace at scaling factors.
Train instances use public seeds (`eval/train/evaluate.py`); the held-out test
uses the same generator with undisclosed seeds, and all its code and data live
in `eval/test/`.

## Experiments and metrics

| experiment | setting | metrics (coefficient) |
|---|---|---|
| `qwen_peak_goodput` (weight 0.5) | TTFT 500 / TPOT 50 ms; loads 1.0–5.0 RPS (step 0.5), 6 instances per load | `peak_goodput_gain_pct` (1.0, primary), `mean_goodput_gain_pct` (0.5) |
| `qwen_latency_detail` (weight 0.25) | TTFT 500 / TPOT 50 ms; 2.5 RPS, 12 instances | `ttft_p99_ms` (1.0, primary), `slo_attainment` (1.0), `tpot_p99_ms` (0.5), `ttft_p50_ms` (0.25) |
| `qwen_relaxed_slo_goodput` (weight 0.25) | TTFT 1000 / TPOT 100 ms; loads 1.5–6.0 RPS (step 0.5), 6 instances per load | `peak_goodput_gain_pct` (1.0, primary), `mean_goodput_gain_pct` (0.5) |

- **effective RPS** at a load = offered RPS × fraction of requests that meet
  both SLOs (rejected and unfinished requests fail), averaged over the
  instances of that load.  `peak_*` is its maximum over the load sweep,
  `mean_*` its mean over the sweep; both are reported as % gain over the
  better of the evaluator's own frozen copies of the two baselines on the same
  instances (so editing `baseline/` cannot move the comparison point).
- **slo_attainment**: fraction of all offered requests meeting both SLOs.
- **TTFT / TPOT percentiles** are over admitted requests (an admitted request
  that never completes counts at 20× the SLO).  A request's TPOT is its worst
  per-token rate `max_j (t_j − max(t_0, arrival + ttft_slo)) / j`, i.e. the
  per-token rate measured from the later of its first token and its TTFT
  deadline.
- **Gate:** if `slo_attainment` falls below that of the weakest evaluator
  baseline, every metric of `qwen_latency_detail` takes its worst value.
- **Failure:** a method whose run fails anywhere gets the worst value of every
  metric in every experiment (gain −100%, attainment 0, TTFT 10 000 ms, TPOT
  1 000 ms).

Everything is recomputed by the evaluator from its own simulation; nothing a
scheduler prints other than its decisions is read.  Results are
deterministic: the same scheduler on the same instances gives identical
numbers.

## Commands

- `scripts/method.sh`, `scripts/baseline.sh`: register the shipped schedulers.
- `scripts/evaluate_train.sh`: public train split → `scoring/scores_train.json`
  (`python eval/train/evaluate.py --experiments qwen_peak_goodput --log-dir
  tmp/logs` for a faster subset with scheduler stderr logs).
- `scripts/evaluate_test.sh` (no arguments): held-out split →
  `scoring/scores_test.json`.  Only the `results` entries are ranked.
- `scripts/reproduce.sh`: register + train + test end to end (CPU only).
