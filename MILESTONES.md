# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-31 09:45] - method_runs
- Continued from a reset audit one level below the previous `core_claim`.
- Confirmed the method code still implements FairBatching's actual scheduler
  mechanics rather than a surrogate: envelope deadlines, slack tracking,
  adaptive time budget, linear step-time estimation, three-stage packing, and
  PAB admission estimation.
- Fixed effective RPS to use offered trace request rate times SLO satisfaction,
  matching the paper's goodput definition and avoiding inflated PAB scores when
  requests are rejected.
- Fixed standalone flat-output evaluation so custom traces are not mislabeled
  as `single_node_qwen_trace`.
- Local `scripts/reproduce.sh "" outputs && python validate.py --compare`
  passes after the fixes; submitting a CPU container job before restoring
  `core_claim`.

### [2026-05-31 09:24] - core_claim
- Implemented FairBatching's envelope-deadline scheduler, adaptive time budget,
  linear batch-time model, three-stage batch packing, and PAB admission estimate
  in a trace-driven simulator.
- Verified the packaged reproduction with `scripts/reproduce.sh "" outputs` and
  `python validate.py --compare`.
- On the QwenTrace-like core experiment, FairBatching improves effective RPS
  over both baselines while keeping the envelope TPOT constraint near 50 ms and
  eliminating SLO violations.

### [2026-05-31 09:28] - core_claim
- CPU container job `1bb3aa42-356` completed with exit code 0.
- The job regenerated the synthetic traces, ran baselines and FairBatching
  variants, wrote `scoring/scores.json`, and passed `python validate.py
  --compare` inside the packaged container.

### [2026-05-31 09:32] - core_claim
- Re-audited the implementation against the paper's algorithm and confirmed the
  evaluator reads scheduler artifacts without importing `method/`.
- Kept the milestone at `core_claim`: the main QwenTrace-like result supports
  the central single-node claim, while the BurstGPT-like trace remains a
  non-discriminative portability check rather than a stronger claim.

## Stop Justification
- Not stopped. A CPU verification job is being submitted for the audited fixes.
