#!/usr/bin/env bash
set -euo pipefail
cd /home/user
python -m redteam.run_public_selection
python -m redteam.analyze
