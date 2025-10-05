#!/usr/bin/env bash
set -euo pipefail

# Flexible build script
# Examples:
#   ./build.sh                      # configure + build default target in build/release
#   ./build.sh -t tensor_ops_test   # build only the tensor_ops_test target
#   ./build.sh -B build/debug -T Debug -- -DENABLE_TESTS=ON

detect_apple_mcpu() {
  if [[ -n "${APPLE_MCPU_OVERRIDE:-}" ]]; then
    printf '%s' "${APPLE_MCPU_OVERRIDE}"
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    local brand
    brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf '')
    if [[ $brand =~ Apple[[:space:]]+(M[0-9]+) ]]; then
      local chip
      chip=$(printf '%s' "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')
      printf '%s' "-mcpu=apple-${chip}"
      return
    fi
  fi
  printf '%s' "-mcpu=apple-m1"
}

BUILD_DIR=${BUILD_DIR:-build/release}
BUILD_TYPE=${BUILD_TYPE:-Release}
CXX=${CXX:-clang++}
TARGET=""
EXTRA_ARGS=()

# VECTOR_DIAGNOSTIC_FLAGS=${VECTOR_DIAGNOSTIC_FLAGS:-"-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"}
VECTOR_DIAGNOSTIC_FLAGS=
CPU_FLAG=$(detect_apple_mcpu)
DEFAULT_CXX_FLAGS="${CXXFLAGS:-} -O3 ${CPU_FLAG} ${VECTOR_DIAGNOSTIC_FLAGS}"

while (( "$#" )); do
  case "$1" in
    -B|--build-dir)
      BUILD_DIR="$2"; shift 2;;
    -T|--type)
      BUILD_TYPE="$2"; shift 2;;
    -t|--target)
      TARGET="$2"; shift 2;;
    --)
      shift; EXTRA_ARGS+=("$@"); break;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="${DEFAULT_CXX_FLAGS}" \
  "${EXTRA_ARGS[@]:-}"

if [[ -n "$TARGET" ]]; then
  cmake --build "$BUILD_DIR" --target "$TARGET" -- -j2
else
  cmake --build "$BUILD_DIR" -- -j2
fi

# convenience symlink for tooling
rm -f compile_commands.json || true
ln -s "$BUILD_DIR/compile_commands.json" compile_commands.json || true
