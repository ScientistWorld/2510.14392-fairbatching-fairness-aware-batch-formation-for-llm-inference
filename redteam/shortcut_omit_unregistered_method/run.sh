#!/usr/bin/env bash
set -euo pipefail
cd /home/user
python -m redteam.run_case shortcut_omit_unregistered_method
