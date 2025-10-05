#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" == "t" ]; then
    ./test.sh
    exit $?
fi

./build.sh
time ./build/release/tformer
if [ -d .venv ]; then
  # optional: activate python venv for vis scripts
  source .venv/bin/activate || true
fi
if [ "${1:-}" == "v" ]; then
    python vis_loss.py
elif [ "${1:-}" == "v2" ]; then
    python vis_loss_v2.py
elif [ "${1:-}" == "v3" ]; then
    python vis_loss_v3.py
fi
