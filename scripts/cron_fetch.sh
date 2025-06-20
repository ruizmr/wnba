#!/usr/bin/env bash
# Run nightly_fetch.py in Ray job.  Intended for cron.
set -euo pipefail

# Address of Ray head. When running on cluster, autoscaler sets RAY_ADDRESS.
RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}

ray job submit \
  --address "$RAY_ADDRESS" \
  -- python -m data.nightly_fetch "$@"