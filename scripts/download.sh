#!/usr/bin/env bash
# Estimated download size: 0 bytes. This reproduction uses tracked synthetic
# traces generated from the paper's reported aggregate trace characteristics.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/downloads
touch data/downloads/.gitkeep
python data/workloads.py --name qwen_synthetic --output data/qwen_synthetic.csv --rps 2.4
python data/workloads.py --name burstgpt_synthetic --output data/burstgpt_synthetic.csv
python data/workloads.py --name decode_slack_burst --output data/decode_slack_burst.csv
