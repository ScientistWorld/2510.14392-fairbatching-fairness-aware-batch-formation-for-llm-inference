#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
EXPERIMENT_NAME="${2:-}"
if [[ -n "$EXPERIMENT_NAME" ]]; then
    python eval/test/evaluate.py --outputs "$OUTDIR" --scores scoring/scores_test.json --experiment-name "$EXPERIMENT_NAME"
else
    python eval/test/evaluate.py --outputs "$OUTDIR" --scores scoring/scores_test.json
fi
