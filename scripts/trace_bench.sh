#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/trace_bench.sh <mode> [target] [args...]
  mode   : t (Time Profiler) | m (Allocations)
  target : optional. If it resolves to an executable (absolute, relative,
           or within ./build/debug/), that binary is traced directly.
           Otherwise the arguments are passed to ./build/debug/tformer.
Examples:
  ./scripts/trace_bench.sh t
  ./scripts/trace_bench.sh t mnist
  ./scripts/trace_bench.sh t microbenchmarks/mnist_csv/mnist_csv_bench_main
  ./scripts/trace_bench.sh m ./build/debug/microbenchmarks/foo_bench arg1 arg2
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift || true

if [[ "$MODE" != "t" && "$MODE" != "m" ]]; then
  usage
  exit 1
fi

./scripts/build_debug.sh

DEFAULT_EXEC="./build/debug/microbenchmarks/mnist_csv/mnist_csv_bench_main"
TRACE_BIN="$DEFAULT_EXEC"
TRACE_ARGS=()

if [[ $# -gt 0 ]]; then
  TARGET_SPEC="$1"
  shift || true

  if [[ -x "$TARGET_SPEC" ]]; then
    TRACE_BIN="$TARGET_SPEC"
    TRACE_ARGS=("$@")
  elif [[ -x "./build/debug/$TARGET_SPEC" ]]; then
    TRACE_BIN="./build/debug/$TARGET_SPEC"
    TRACE_ARGS=("$@")
  else
    TRACE_BIN="./build/debug/tformer"
    TRACE_ARGS=("$TARGET_SPEC" "$@")
  fi
else
  if [[ ! -x "$DEFAULT_EXEC" ]]; then
    TRACE_BIN="./build/debug/tformer"
    TRACE_ARGS=("mnist")
  fi
fi

TRACE_DIR="$(cd "$(dirname "$TRACE_BIN")" && pwd)"
TRACE_BIN="${TRACE_DIR}/$(basename "$TRACE_BIN")"

TRACE_OUT="tformer_trace.trace"
DSYM_PATH="${TRACE_BIN}.dSYM"

echo "Tracing binary : $TRACE_BIN"
if [[ ${#TRACE_ARGS[@]} -gt 0 ]]; then
  printf "With arguments: %s\n" "${TRACE_ARGS[*]}"
fi

sudo rm -rf "$TRACE_OUT" "$DSYM_PATH"

xcrun dsymutil "$TRACE_BIN" -o "$DSYM_PATH"
if [[ -f entitlements.plist ]]; then
  if ! codesign --entitlements entitlements.plist --sign - --force "$TRACE_BIN" 2>/tmp/trace_codesign.err; then
    if grep -q "unused entitlement" /tmp/trace_codesign.err; then
      codesign --sign - --force "$TRACE_BIN"
    else
      cat /tmp/trace_codesign.err
      rm -f /tmp/trace_codesign.err
      exit 1
    fi
  fi
  rm -f /tmp/trace_codesign.err
else
  codesign --sign - --force "$TRACE_BIN"
fi

TRACE_TEMPLATE="Time Profiler"
XC_CMD=(xctrace record --template "$TRACE_TEMPLATE" --output "$TRACE_OUT" --launch "$TRACE_BIN")
if [[ "$MODE" == "m" ]]; then
  TRACE_TEMPLATE="Allocations"
  XC_CMD=(sudo xctrace record --template "$TRACE_TEMPLATE" --output "$TRACE_OUT" --launch "$TRACE_BIN")
fi

if [[ ${#TRACE_ARGS[@]} -gt 0 ]]; then
  XC_CMD+=("${TRACE_ARGS[@]}")
fi

"${XC_CMD[@]}"

sudo chmod -R a+rX "$TRACE_OUT"
sudo open "$TRACE_OUT"
