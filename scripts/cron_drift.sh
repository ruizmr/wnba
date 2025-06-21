#!/usr/bin/env bash
# Run data drift check in a non-interactive cron context
set -euo pipefail

export WEHOOP_DATA_PATH="${WEHOOP_DATA_PATH:-/data/wehoop/wnba}"

# Activate virtualenv if exists
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

python -m python.monitor.drift_check | cat