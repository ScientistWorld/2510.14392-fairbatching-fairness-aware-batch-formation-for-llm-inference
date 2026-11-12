#!/usr/bin/env bash
# Register the paper's schedulers.  A method is a program that speaks the
# scheduler protocol (briefing/evaluation.md); its manifest under
# outputs/schedulers/ tells the evaluators how to launch it.
set -euo pipefail

cd "$(dirname "$0")/.."
OUTDIR="${1:-outputs}"
mkdir -p "$OUTDIR/schedulers"
cat > "$OUTDIR/schedulers/fairbatching.json" <<'JSON'
{"command": ["python", "method/fairbatching.py", "--variant", "vanilla"]}
JSON
cat > "$OUTDIR/schedulers/fairbatching_pab.json" <<'JSON'
{"command": ["python", "method/fairbatching.py", "--variant", "pab"]}
JSON
