# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 00:00] - none
- Started reproduction work.
- Read the paper and identified FairBatching's scheduler as the core algorithm.

### [2026-05-31 00:00] - method_runs
- Implemented FairBatching's envelope-deadline slack tracking, adaptive time budget, three-stage batch construction, and PAB estimate.
- Implemented prefill-priority and Sarathi-style baselines.
- End-to-end scripts run and produce `scoring/scores.json`.

### [2026-05-31 00:00] - core_claim
- Fixed the FairBatching urgent-decode threshold to match Algorithm 1.
- Evaluated a scaled QwenTrace-like load point where FairBatching improves effective RPS over both baselines while satisfying envelope SLOs.
- Fixed the compute-container baseline import path after job `73b67506-68e` failed before running experiments.

### [2026-05-31 09:22] - core_claim
- Retry job `048586e6-92a` completed with exit code 0.
- The packaged `scripts/reproduce.sh` workflow reran baselines, FairBatching, evaluation, and `validate.py --compare` successfully.
- Final reproduced core-claim result: FairBatching effective RPS 2.144 with 0.000 SLO violation versus vanilla effective RPS 1.623 with 0.244 SLO violation on the scaled QwenTrace-like workload.

## Stop Justification
- Completed at milestone `core_claim`.
