#!/usr/bin/env bash
# Reproduce the reference end to end (CPU only, no internet needed):
# registers the paper's schedulers and baselines, scores them on the public
# train workloads (scoring/scores_train.json, scoring/scores.json) and on the
# held-out workloads (scoring/scores_test.json), then validates the workspace.
#
# The iteration-11 hardening audit (variants and shortcuts through the same
# evaluators) is `python redteam/hardening/run_audit.py`; it resumes from
# redteam/hardening/runs/ and is documented in HARDENING.md.
set -euo pipefail

cd /home/user
rm -rf outputs
scripts/reproduce.sh outputs
python validate.py
