#!/usr/bin/env bash
set -euo pipefail

# Configure, build and run tests with CMake/CTest
# Uses clang++ if available, otherwise default C++ compiler

BUILD_DIR="${BUILD_DIR:-build/release}"
CXX=${CXX:-clang++}

extra_cmake_args=(
  "-DENABLE_TESTS=ON"
  "-DCMAKE_CXX_COMPILER=${CXX}"
)

if command -v ccache >/dev/null 2>&1; then
  export CCACHE_BASEDIR="${CCACHE_BASEDIR:-$(pwd)}"
  export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-2G}"
  extra_cmake_args+=("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
fi

./build.sh -B "$BUILD_DIR" -T Release -- "${extra_cmake_args[@]}"
ctest --test-dir "$BUILD_DIR/tests" --output-on-failure
