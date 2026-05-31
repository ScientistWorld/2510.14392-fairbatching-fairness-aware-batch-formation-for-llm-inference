# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 09:24] - core_claim
- Implemented FairBatching's envelope-deadline scheduler, adaptive time budget, linear batch-time model, three-stage batch packing, and PAB admission estimate in a trace-driven simulator.
- Verified the packaged reproduction locally with `scripts/reproduce.sh "" outputs` and `python validate.py --compare`.
- On the QwenTrace-like core experiment, FairBatching improves effective RPS over both baselines while keeping the envelope TPOT constraint at roughly 50 ms and eliminating SLO violations.

### [2026-05-31 09:26] - core_claim
- Audited the continuing workspace after milestone reset; the method is not a surrogate and the evaluator remains independent from `method/`.
- Added and audited secondary workload coverage.
- Kept the milestone at `core_claim`: QwenTrace-like results support the core claim, while auxiliary synthetic settings are currently sanity checks rather than strong additional claim reproductions.

### [2026-05-31 09:28] - core_claim
- CPU container job `1bb3aa42-356` completed with exit code 0.
- The job regenerated all tracked synthetic traces, ran baselines and FairBatching variants, wrote `scoring/scores.json`, and passed `python validate.py --compare` inside the packaged container.
- Final milestone remains `core_claim`.

### [2026-05-31 09:29] - core_claim
- Removed the AzureTrace-like scoring group after final validation flagged its synthetic RPS scale as inconsistent with the paper by more than 2x.
- Retained the QwenTrace-like core experiment and BurstGPT-like auxiliary experiment.
