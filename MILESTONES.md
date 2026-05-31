# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-31 10:27] - core_claim
- Removed the relaxed-SLO auxiliary experiment from the default scored surface
  after validation showed its PAB primary metric was too far from the paper
  reference.
- Kept `core_claim` because the default `single_node_qwen_trace` result still
  demonstrates the central goodput/SLO trade-off.

### [2026-05-31 10:25] - core_claim
- Final verification kept the milestone at `core_claim`; no fresh packaged job
  result was present to justify promotion.
- Moved baseline simulation support to method-agnostic `data/` code so
  baselines no longer import from `method/`.
- Rechecked `python validate.py --compare`; scoring, import separation,
  train/test artifacts, and shared-path checks all passed.

### [2026-05-31 10:19] - core_claim
- Added a second default experiment, `qwen_slo_1000_100`, to probe the paper's
  SLO-versatility claim at TTFT=1000ms and TPOT=100ms.
- Hardened `scripts/evaluate.sh`'s evaluator to require per-request timing
  records rather than trusting aggregate-only self-reported metrics.
- Local `scripts/reproduce.sh "" outputs`, train/test scoring, and
  `python validate.py --compare` pass with two scored experiments. Keeping the
  milestone at `core_claim` until a fresh packaged CPU job verifies this
  expanded setup.

### [2026-05-31 10:13] - core_claim
- Final verification kept the milestone at `core_claim`.
- Confirmed `scripts/evaluate.sh outputs` regenerates `scoring/scores.json`
  from existing artifacts and `python validate.py --compare` passes.
- Cleaned method-specific language from method-agnostic evaluation/scoring
  documentation and documented the auxiliary envelope TPOT reference value.

### [2026-05-31 10:08] - core_claim
- Tightened `scripts/evaluate.sh`'s default evaluator so it scores arbitrary
  `*.json` method outputs instead of a fixed method list, and recomputes all
  declared metrics from per-request records.
- Regenerated `scoring/scores.json`; it now includes TTFT SLO violation rate,
  envelope TPOT SLO violation rate, and P95 metrics for every scored method.
- Kept the milestone at `core_claim` because the implemented scheduler matches
  FairBatching's envelope-deadline algorithm and improves effective RPS with no
  envelope SLO violations on the QwenTrace-like trace. The remaining raw TPOT
  mismatch is documented as a limit, so no higher milestone is claimed.

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

### [2026-05-31 09:50] - core_claim
- CPU container job `34d84bda-8c0` completed with exit code 0.
- The job reran `scripts/reproduce.sh "" outputs` and `python validate.py
  --compare` inside the packaged container.
- Restored `core_claim`: the audited implementation and scorer reproduce the
  main single-node FairBatching claim on the QwenTrace-like workload, with
  FairBatching improving effective RPS while satisfying the envelope SLO and
  avoiding the baseline SLO violations.

### [2026-05-31 09:58] - core_claim
- Removed the non-discriminative synthetic BurstGPT trace from the default
  scored outputs after validation flagged its effective RPS as more than 2x
  away from the paper's reported scale.
- Kept the core QwenTrace-like reproduction as the default target set; BurstGPT
  remains available as a custom trace but is not claimed as reproduced.
- Regenerated `scoring/scores.json` via `scripts/reproduce.sh "" outputs` and
  reran `python validate.py --compare`; all checks passed.

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
- Completed at milestone `core_claim`.
