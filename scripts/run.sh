#!/usr/bin/env bash
set -euo pipefail

cd /home/user
echo "cpus: $(nproc)"
python --version

# Reference results (what every future attempt is ranked against).
rm -rf outputs
scripts/reproduce.sh outputs

# Iteration-11 hardening audit: variants and shortcuts through the evaluators.
python redteam/hardening/run_audit.py

python validate.py
