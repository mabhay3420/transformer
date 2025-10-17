#!/usr/bin/env bash
set -euo pipefail

target="xor"
vis_mode=""

for arg in "$@"; do
  case "$arg" in
    t)
      ./scripts/test.sh
      exit $?
      ;;
    v|v2|v3)
      vis_mode="$arg"
      ;;
    *)
      target="$arg"
      ;;
  esac
done

./scripts/build.sh
time ./build/release/tformer "$target"
if [ -d .venv ]; then
  # optional: activate python venv for vis scripts
  source .venv/bin/activate || true
fi

case "$vis_mode" in
  v)
    python scripts/vis_loss.py
    ;;
  v2)
    python scripts/vis_loss_v2.py
    ;;
  v3)
    python scripts/vis_loss_v3.py
    ;;
esac
