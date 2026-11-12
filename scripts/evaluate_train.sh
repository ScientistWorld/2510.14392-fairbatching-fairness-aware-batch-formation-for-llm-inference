#!/usr/bin/env bash
# Score every scheduler registered under outputs/schedulers/ on the public
# train workloads; writes scoring/scores_train.json.
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
python eval/train/evaluate.py --outputs "$OUTDIR" --scores scoring/scores_train.json
