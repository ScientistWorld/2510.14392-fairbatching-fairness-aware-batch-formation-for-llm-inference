#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
EXPERIMENT_NAME="${2:-}"
if [[ -n "$EXPERIMENT_NAME" ]]; then
    python eval/train/evaluate.py --outputs "$OUTDIR" --scores scoring/scores_train.json --experiment-name "$EXPERIMENT_NAME"
else
    python eval/train/evaluate.py --outputs "$OUTDIR" --scores scoring/scores_train.json
fi
