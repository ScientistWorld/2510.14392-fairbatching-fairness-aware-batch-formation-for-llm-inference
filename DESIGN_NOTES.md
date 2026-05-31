# Design Notes

Checked the train/test split, wrapper scripts, scoring schema, and paper context
for the FairBatching scheduler benchmark.

Changes made:

- Fixed a gameability issue in `eval/train/evaluate.py` and
  `eval/test/evaluate.py`: both slices now recompute SLO success from
  per-request `ttft_ms`, `max_envelope_tpot_ms`, and per-request SLO fields
  instead of trusting the method-emitted `slo_met` summary flag.
- Reduced trust in method summary fields by recomputing offered request rate
  from per-request arrival times when available, instead of using only the
  method-emitted `offered_rps`.
- Added separate `ttft_slo_violation_rate` and
  `envelope_tpot_slo_violation_rate` metrics so a method cannot hide whether it
  gained goodput by sacrificing first-token latency or envelope TPOT.
- Added P95 tail metrics for TTFT, raw TPOT, and envelope TPOT alongside the
  existing P99 metrics. This gives the train/test split more tail-shape
  triangulation and makes single-threshold tuning easier to detect.
- Updated `scoring/reference.json` so all active metrics have
  `higher_is_better`, `coefficient`, and `description`, and so train/test
  evaluators retain the newly computed metrics instead of filtering them out.

Checks performed:

- `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` pass `bash -n`.
- `evaluate_test.sh` references `eval/test/` and writes
  `scoring/scores_test.json`; `evaluate_train.sh` references `eval/train/` and
  writes `scoring/scores_train.json`.
- No `scoring/scores_train.json` or `scoring/scores_test.json` stub exists.
- Grepped for train/test cross-imports. Only explanatory docstring mentions
  were found, not imports.
- Grepped for forbidden paper-agreement metric names. No metric names match the
  forbidden patterns.

Runtime concerns for the runner iteration:

- The split still scores method-authored per-request latency records. This is
  appropriate for a method-agnostic artifact evaluator, but a malicious method
  can fabricate records unless the runner later cross-checks record IDs,
  arrivals, and SLO fields against the trace named in the artifact.
- Runner update: `single_node_burstgpt_trace` was executed and all methods met
  every SLO on both train and test, leaving goodput and violation metrics
  saturated. It was removed from active reference scoring; the active
  train/test target is the non-saturated QwenTrace split.
- The first runner job regenerated both QwenTrace and BurstGPT artifacts, then
  wrote `scoring/scores_train.json` and `scoring/scores_test.json`. The
  follow-up job overwrote those files with active QwenTrace scores only.
- Final train/test check: both slices score the same four method identifiers on
  the same ten metrics. QwenTrace train/test goodput is stable for the proposed
  method, while baseline SLO and tail-latency values differ enough to confirm
  the held-out split is not a copy of the visible slice.
- Validation follow-up: split effective RPS is capped by the recomputed
  whole-trace success fraction so a lucky hash bucket cannot report higher
  capacity than the method achieved on the full trace.
