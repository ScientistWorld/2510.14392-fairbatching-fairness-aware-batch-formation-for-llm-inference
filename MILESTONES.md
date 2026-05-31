# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-31 00:00] - none
- Started reproduction work.
- Read the paper and identified FairBatching's scheduler as the core algorithm.

### [2026-05-31 00:00] - method_runs
- Implemented FairBatching's envelope-deadline slack tracking, adaptive time budget, three-stage batch construction, and PAB estimate.
- Implemented prefill-priority and Sarathi-style baselines.
- End-to-end scripts run and produce `scoring/scores.json`.

## Stop Justification
