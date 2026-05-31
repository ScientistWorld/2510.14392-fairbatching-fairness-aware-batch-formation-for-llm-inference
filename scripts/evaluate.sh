#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
EXPERIMENT_NAME="${2:-}"
if [[ -n "$EXPERIMENT_NAME" ]]; then
    python eval/evaluate_outputs.py --outputs "$OUTDIR" --scores scoring/scores.json --experiment-name "$EXPERIMENT_NAME"
else
    python eval/evaluate_outputs.py --outputs "$OUTDIR" --scores scoring/scores.json
fi
