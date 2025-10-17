1. To build the code: `./scripts/build.sh`
2. To run the code e2e: `./scripts/run.sh`

For debug and address sanitized versions:

1. To build the code: `./scripts/build_debug.sh`
2. To run the code e2e: `./scripts/run_debug.sh`

## Testing Guidelines

- Always add unit tests for new ops, layers, and training loops under `tests/` using GoogleTest.
- Run tests using `./scripts/test.sh`.
- After significant refactors or feature work, run at least one end-to-end workflow via `./scripts/run.sh <command>` (e.g. `./scripts/run.sh mnist`) to guard against integration regressions.
- Keep tests small and fast; prefer simple shape cases and exact gradient checks (or finite-difference checks) for new tensor ops.
- Run tests whenever making changes to core autograd/memory/tensor code.

## Tools Usage
1. `ast-grep`: A tool for searching and replacing text in source code.

## Python Environment
We use `.venv` and `uv pip install` for installing dependencies.

## Benchmarks

- Prefer adding or extending microbenchmarks under `microbenchmarks/` when
  investigating performance. Use `./scripts/bench.sh <name> [NAME=value ...]` to build
  and run them (e.g. `./scripts/bench.sh matmul M=256 K=256 N=256`).
- Running `./scripts/bench.sh matmul` without extra arguments executes a default suite
  (skinny/square, small/large) illustrating where each implementation wins.
- Keep hot paths simple: prefer straightforward control flow (e.g. `if/else`
  blocks) over lambdas or std::function captures inside tight loops unless
  absolutely necessary for clarity.
- When asked to create or run benchmarks without changing runtime behavior,
  focus on instrumentation only: scaffold the benchmark, wire it into the
  registry, and document usage. Do **not** introduce algorithmic optimizations
  unless explicitly requested.
- Capture sample benchmark output in the task results when relevant, especially
  for multiple shapes or iteration counts.
