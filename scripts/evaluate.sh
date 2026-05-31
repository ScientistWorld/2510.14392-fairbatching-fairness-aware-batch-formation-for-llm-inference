#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
python eval/evaluate_outputs.py --outputs "$OUTDIR" --scores scoring/scores.json
