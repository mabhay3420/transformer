#!/usr/bin/env bash
set -euo pipefail

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

BUILD_DIR=${BUILD_DIR:-build}
CXX=${CXX:-clang++}
# VECTOR_DIAGNOSTIC_FLAGS=${VECTOR_DIAGNOSTIC_FLAGS:-"-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"}
VECTOR_DIAGNOSTIC_FLAGS=
CPU_FLAG=$(detect_apple_mcpu)
DEFAULT_CXX_FLAGS="${CXXFLAGS:-} ${CPU_FLAG} ${VECTOR_DIAGNOSTIC_FLAGS}"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="${DEFAULT_CXX_FLAGS}"

cmake --build "$BUILD_DIR"
