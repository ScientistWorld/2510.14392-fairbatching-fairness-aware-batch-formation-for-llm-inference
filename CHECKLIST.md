# Reproduction Checklist

## Briefing
- [x] `briefing/problem.md`, `briefing/evaluation.md`, `briefing/method.md`, `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json` (3 experiments; paper numbers unchanged)
- [x] `scoring/scores.json`, `scores_train.json`, `scores_test.json` from the reference runs
- [x] `scoring/EXPERIMENTS.md`, `TARGETS.md`, `CONSTRAINTS.md`, `DIRECTION.md`
- [x] `python validate.py` passes

## Code
- [x] `eval/train/`, `eval/test/` (self-contained held-out evaluator)
- [x] `method/fairbatching.py`, `baseline/schedulers.py`
- [x] `data/` simulator, protocol helper, QwenTrace head segment

## Scripts / environment
- [x] `scripts/download.sh`, `reproduce.sh`, `run.sh`, `method.sh`, `baseline.sh`, `evaluate*.sh`
- [x] `environment/container.def` (python:3.11-slim, standard library only)

## Hardening
- [x] `HARDENING.md`, audit in `redteam/hardening/`

## Milestones
- [x] `method_runs`, `core_claim`, `core_claim_plus`
- [ ] cluster-level PAB load balancing, BurstGPT / Azure traces, other models
