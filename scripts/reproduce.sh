#!/usr/bin/env bash
# Reproduce the reference results end to end (CPU only, no internet needed:
# the derived workload table data/qwen_trace_head.csv is tracked in git).
#   1. register the paper's schedulers and the baselines under outputs/;
#   2. score them on the public train workloads  -> scoring/scores_train.json
#      (also copied to scoring/scores.json);
#   3. score them on the held-out workloads      -> scoring/scores_test.json.
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
rm -rf "$OUTDIR/schedulers"
scripts/baseline.sh "$OUTDIR"
scripts/method.sh "$OUTDIR"
scripts/evaluate_train.sh "$OUTDIR"
cp scoring/scores_train.json scoring/scores.json
scripts/evaluate_test.sh
