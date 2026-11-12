#!/usr/bin/env bash
set -euo pipefail

cd /home/user

rm -rf outputs
scripts/reproduce.sh "" outputs

bash scripts/evaluate_train.sh outputs
bash scripts/evaluate_test.sh outputs
python validate.py --compare
