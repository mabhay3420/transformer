#!/usr/bin/env bash
set -euo pipefail

TARGET=""
if [ $# -lt 1 ]; then
  echo "Usage: $0 <benchmark> [args]"
  exit 1
fi

TARGET="$1"
shift

./scripts/build.sh

case "$TARGET" in
  matmul)
    ./build/release/microbenchmarks/matmul/matmul_bench_main "$@"
    ;;
  elementwise)
    ./build/release/microbenchmarks/elementwise/elementwise_bench_main "$@"
    ;;
  csv_loader)
    ./build/release/microbenchmarks/csv_loader/csv_loader_bench_main "$@"
    ;;
  *)
    echo "Unknown benchmark: $TARGET"
    exit 1
    ;;
esac
