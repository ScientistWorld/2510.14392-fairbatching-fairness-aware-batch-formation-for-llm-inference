# Reproduction Checklist

## Briefing
- [x] `briefing/problem.md`
- [x] `briefing/evaluation.md`
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json`
- [x] Workspace validated via `python validate.py --compare`
- [x] `scoring/scores.json`
- [x] `scoring/EXPERIMENTS.md`
- [x] `scoring/TARGETS.md`
- [x] `scoring/CONSTRAINTS.md`
- [x] `scoring/DIRECTION.md`

## Code
- [x] Evaluation code in `eval/`
- [x] Method implementation in `method/`
- [x] Baseline implementation in `baseline/`

## Scripts
- [x] `scripts/evaluate.sh`
- [x] `scripts/reproduce.sh`
- [x] `scripts/download.sh`
- [x] `scripts/baseline.sh`

## Environment
- [x] `environment/container.def` + `environment/setup.sh`

## Data
- [x] Synthetic trace generator in `data/`
- [x] QwenTrace-like scored trace
- [x] BurstGPT-like custom trace generator retained but not scored by default
- [x] No pretrained models needed

## Milestones
- [x] `method_runs`
- [x] `core_claim`
- [ ] `core_claim_plus`
- [ ] `secondary_claims`
- [ ] `majority`
- [ ] `near_complete`
- [ ] `full`
