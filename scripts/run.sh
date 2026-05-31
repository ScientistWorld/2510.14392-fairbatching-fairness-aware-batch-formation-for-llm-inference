#!/usr/bin/env bash
set -euo pipefail

cd /home/user
rm -rf outputs
mkdir -p outputs/single_node_qwen_trace

scripts/baseline.sh data/qwen_synthetic.csv outputs/single_node_qwen_trace
scripts/method.sh data/qwen_synthetic.csv outputs/single_node_qwen_trace

bash scripts/evaluate_train.sh outputs
bash scripts/evaluate_test.sh outputs
