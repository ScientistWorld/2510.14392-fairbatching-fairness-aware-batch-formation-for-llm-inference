#!/usr/bin/env bash
# Register the paper's baseline schedulers (vLLM-vanilla, vLLM-sarathi).
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
mkdir -p "$OUTDIR/schedulers"
cat > "$OUTDIR/schedulers/vllm_vanilla.json" <<'JSON'
{"command": ["python", "baseline/schedulers.py", "--policy", "vllm_vanilla"]}
JSON
cat > "$OUTDIR/schedulers/vllm_sarathi.json" <<'JSON'
{"command": ["python", "baseline/schedulers.py", "--policy", "vllm_sarathi"]}
JSON
