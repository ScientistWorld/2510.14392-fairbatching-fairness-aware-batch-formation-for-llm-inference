#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
TRACE="${1:-data/qwen_synthetic.csv}"
OUTDIR="${2:-outputs}"
mkdir -p "$OUTDIR"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python baseline/run_baseline.py --trace "$TRACE" --policy vllm_vanilla --output "$OUTDIR/vllm_vanilla.json"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python baseline/run_baseline.py --trace "$TRACE" --policy vllm_sarathi --output "$OUTDIR/vllm_sarathi.json"
