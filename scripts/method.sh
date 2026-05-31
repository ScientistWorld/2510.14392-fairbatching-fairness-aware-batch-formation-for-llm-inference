#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
TRACE="${1:-data/qwen_synthetic.csv}"
OUTDIR="${2:-outputs}"
mkdir -p "$OUTDIR"
python method/simulator.py --trace "$TRACE" --policy fairbatching --output "$OUTDIR/fairbatching.json"
python method/simulator.py --trace "$TRACE" --policy fairbatching_pab --output "$OUTDIR/fairbatching_pab.json"
