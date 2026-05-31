#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
TRACE="${1:-data/qwen_synthetic.csv}"
OUTDIR="${2:-outputs}"

scripts/download.sh
scripts/baseline.sh "$TRACE" "$OUTDIR"
scripts/method.sh "$TRACE" "$OUTDIR"
scripts/evaluate.sh "$OUTDIR"
