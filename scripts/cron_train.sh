#!/usr/bin/env bash
# Trigger training sweep via Ray Train + Tune.
set -euo pipefail

RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}

ray job submit \
  --address "$RAY_ADDRESS" \
  -- python -m model.train --smoke-test "$@"