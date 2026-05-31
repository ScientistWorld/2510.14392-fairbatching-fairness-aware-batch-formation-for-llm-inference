#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
TRACE="${1:-}"
OUTDIR="${2:-outputs}"

scripts/download.sh
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

if [[ -n "$TRACE" ]]; then
    scripts/baseline.sh "$TRACE" "$OUTDIR"
    scripts/method.sh "$TRACE" "$OUTDIR"
    scripts/evaluate.sh "$OUTDIR" "${OUTDIR##*/}"
else
    scripts/baseline.sh data/qwen_synthetic.csv "$OUTDIR/single_node_qwen_trace"
    scripts/method.sh data/qwen_synthetic.csv "$OUTDIR/single_node_qwen_trace"
    scripts/evaluate.sh "$OUTDIR"
fi
