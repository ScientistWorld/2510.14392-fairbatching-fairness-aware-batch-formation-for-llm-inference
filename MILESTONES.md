# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 09:24] - core_claim
- Implemented FairBatching's envelope-deadline scheduler, adaptive time budget, linear batch-time model, three-stage batch packing, and PAB admission estimate in a trace-driven simulator.
- Verified the packaged reproduction locally with `scripts/reproduce.sh "" outputs` and `python validate.py --compare`.
- On the QwenTrace-like core experiment, FairBatching improves effective RPS over both baselines while keeping the envelope TPOT constraint at roughly 50 ms and eliminating SLO violations.
