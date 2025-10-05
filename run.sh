#!/usr/bin/env bash
set -euo pipefail

target="xor-pt"
vis_mode=""

for arg in "$@"; do
  case "$arg" in
    t)
      ./test.sh
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

./build.sh
time ./build/release/tformer "$target"
if [ -d .venv ]; then
  # optional: activate python venv for vis scripts
  source .venv/bin/activate || true
fi

case "$vis_mode" in
  v)
    python vis_loss.py
    ;;
  v2)
    python vis_loss_v2.py
    ;;
  v3)
    python vis_loss_v3.py
    ;;
esac
