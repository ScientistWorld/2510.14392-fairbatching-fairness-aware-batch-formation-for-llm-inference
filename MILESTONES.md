# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 09:24] - core_claim
- Implemented FairBatching's envelope-deadline scheduler, adaptive time budget, linear batch-time model, three-stage batch packing, and PAB admission estimate in a trace-driven simulator.
- Verified the packaged reproduction locally with `scripts/reproduce.sh "" outputs` and `python validate.py --compare`.
- On the QwenTrace-like core experiment, FairBatching improves effective RPS over both baselines while keeping the envelope TPOT constraint at roughly 50 ms and eliminating SLO violations.

### [2026-05-31 09:26] - core_claim
- Audited the continuing workspace after milestone reset; the method is not a surrogate and the evaluator remains independent from `method/`.
- Added an AzureTrace-like secondary workload from the paper's aggregate trace statistics and Table 3 reference values.
- Kept the milestone at `core_claim`: QwenTrace-like results support the core claim, while BurstGPT/Azure synthetic settings are currently sanity checks rather than strong additional claim reproductions.
