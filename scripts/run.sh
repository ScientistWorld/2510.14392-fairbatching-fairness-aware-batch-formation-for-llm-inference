#!/usr/bin/env bash
set -euo pipefail

cd /home/user
scripts/reproduce.sh data/qwen_synthetic.csv outputs
python validate.py --compare
