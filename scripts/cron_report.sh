#!/usr/bin/env bash
# Generate & e-mail the daily edge report.
set -euo pipefail

RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}

ray job submit \
  --address "$RAY_ADDRESS" \
  -- python -m report.daily "$@"