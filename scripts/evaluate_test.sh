#!/usr/bin/env bash
# Held-out evaluation: runs every scheduler registered under
# outputs/schedulers/ on workloads that only eval/test/ can generate and
# writes scoring/scores_test.json.  Takes no arguments.
set -euo pipefail

cd "$(dirname "$0")/.."
python eval/test/evaluate.py
