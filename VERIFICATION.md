> **Superseded (iteration 11):** this file verifies the pre-iteration-11 gym. The current evaluator, experiments and audit are described in `briefing/evaluation.md` and `HARDENING.md`.

# Verification Report

## Initial Assessment

The workspace started from a claimed `core_claim` reproduction with method
outputs, reusable scoring artifacts, and train/test evaluation files already in
place. The milestone was justified by the existing single-node scheduling
result: the paper method improves effective request goodput while preserving
the SLO constraints that the reproduced baselines violate.

I did not find a fresh packaged job result for the latest two-experiment
extension, so I did not promote the milestone beyond `core_claim`.

## Issues Found

- Minor: `baseline/run_baseline.py` imported the simulator from `method/`,
  which weakened problem/solution separation for from-scratch users.
- Minor: several method-agnostic or high-level documents used paper-specific
  deadline/envelope wording where plain SLO terminology was sufficient.
- Residual limitation: the benchmark uses synthetic QwenTrace-like traces and a
  deterministic trace-driven simulator rather than the paper's full vLLM
  deployment.
- Residual limitation: the current outputs demonstrate the core scheduling
  gain, but the PAB-specific gain is not separately reproduced.
- Critical validation issue after the first finalization: the auxiliary
  `qwen_slo_1000_100` scores were too far from the paper's PAB primary metric
  and therefore could not remain in the default scored surface.

## Fixes Applied

- Added `data/serving_sim.py` as a method-agnostic simulation substrate for
  baseline schedulers and updated `baseline/run_baseline.py` so it no longer
  imports from `method/`.
- Reworded `briefing/evaluation.md`, `scoring/TARGETS.md`,
  `scoring/CONSTRAINTS.md`, and metric descriptions in
  `scoring/reference.json` to describe deadline-based SLO checks without
  leaking paper-specific formulation details.
- Removed the non-reproduced relaxed-SLO auxiliary experiment from default
  `scores*.json`, `scripts/reproduce.sh`, `scoring/EXPERIMENTS.md`, and
  `scoring/TARGETS.md`. It remains as exploratory data/reference context only.
- Ran interface checks for the method and baseline entry points.
- Ran `python validate.py --compare`; all scoring, import-separation,
  train/test, and shared-path checks passed.

## Final Judgment

The workspace is ready to serve as a core-claim research gym. From-scratch
agents can use the problem, data, baseline, and evaluator without importing or
seeing method internals. Improve-mode agents have a working paper-method
implementation and scored artifacts to build on. The main limitation is
experimental fidelity and scale, not fabrication.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic trace-driven simulator rather than full vLLM
  deployment; auxiliary PAB-specific gains are not part of default scoring.
