#!/usr/bin/env bash
set -euo pipefail

# Flexible build script
# Examples:
#   ./build.sh                      # configure + build default target in build/release
#   ./build.sh -t tensor_ops_test   # build only the tensor_ops_test target
#   ./build.sh -B build/debug -T Debug -- -DENABLE_TESTS=ON

BUILD_DIR=${BUILD_DIR:-build/release}
BUILD_TYPE=${BUILD_TYPE:-Release}
CXX=${CXX:-clang++}
TARGET=""
EXTRA_ARGS=()

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
  -DCMAKE_CXX_FLAGS="-O3" \
  "${EXTRA_ARGS[@]:-}"

if [[ -n "$TARGET" ]]; then
  cmake --build "$BUILD_DIR" --target "$TARGET" -- -j2
else
  cmake --build "$BUILD_DIR" -- -j2
fi

# convenience symlink for tooling
rm -f compile_commands.json || true
ln -s "$BUILD_DIR/compile_commands.json" compile_commands.json || true
