#!/usr/bin/env bash
# Public evaluation of every registered scheduler (same as the train split);
# writes scoring/scores.json.
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
python eval/train/evaluate.py --outputs "$OUTDIR" --scores scoring/scores.json
