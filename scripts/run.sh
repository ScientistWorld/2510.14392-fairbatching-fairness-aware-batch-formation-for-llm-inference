#!/usr/bin/env bash
# Verification job for the hardened gym.  Every stage is resumable: compute
# nodes may be preempted, so finished stages leave a marker / result file and
# are skipped when the job restarts.
set -euo pipefail

cd /home/user
echo "cpus: $(nproc)"
python --version
M=redteam/hardening/runs
mkdir -p "$M"

# Reference results (what every future attempt is ranked against); same steps
# as scripts/reproduce.sh, split so a preemption does not redo finished work.
if [[ ! -f "$M/reference_train_done" ]]; then
    rm -rf outputs
    scripts/baseline.sh outputs
    scripts/method.sh outputs
    scripts/evaluate_train.sh outputs
    cp scoring/scores_train.json scoring/scores.json
    touch "$M/reference_train_done"
fi
if [[ ! -f "$M/reference_test_done" ]]; then
    scripts/evaluate_test.sh
    touch "$M/reference_test_done"
fi

# Iteration-11 hardening audit: variants and shortcuts through the evaluators.
python redteam/hardening/run_audit.py

python validate.py
