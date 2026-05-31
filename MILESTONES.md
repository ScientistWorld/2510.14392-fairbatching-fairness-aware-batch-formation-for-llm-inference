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

## Stop Justification
