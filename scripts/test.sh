#!/usr/bin/env bash
set -euo pipefail

# Configure, build and run tests with CMake/CTest
# Uses clang++ if available, otherwise default C++ compiler

BUILD_DIR="${BUILD_DIR:-build/release}"
CXX=${CXX:-clang++}

./scripts/build.sh -B "$BUILD_DIR" -T Release -- -DENABLE_TESTS=ON -DCMAKE_CXX_COMPILER="$CXX"
ctest --test-dir "$BUILD_DIR/tests" --output-on-failure
