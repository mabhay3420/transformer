# Resources
1. Dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
2. Video: https://www.youtube.com/watch?v=kCc8FmEb1nY


# Guidelines
where possible:
    - use std library
    - use auto
    - run `./format.sh --check` before committing (the pre-commit hook enforces this automatically whenever you run `git commit`)

# Running

- `./run.sh` builds the project and executes the default `xor-pt` command, which trains the XOR task using the PyTorch-style `nn` modules and tensor autograd.
- `./build/release/tformer --list` shows all available entrypoints. Use `./build/release/tformer <command>` or `./run.sh <command>` to run an alternative experiment (e.g. `bigram`, `mnist`).

# Layout

- `apps/` hosts thin CLI entrypoints (e.g. `apps/main.cpp`) that dispatch into the library.
- `lib/` is organized by concern: `core/` for tensors/autograd, `nn/` for modules, `data/` for loaders, `models/` for experiment logic, `train/` for shared helpers, and `utils/` for miscellaneous support code.
- Public headers remain under `include/`, with new subdirectories like `include/train/` and `include/data/` mirroring the implementation layout.

# Microbenchmarks

Microbenchmarks live under `microbenchmarks/` and can be built and executed via
the `bench.sh` helper script. For example, to compare the registered matrix
multiplication kernels for a specific shape:

```
./bench.sh matmul M=256 K=256 N=256 I=5
```

Invoking `./bench.sh matmul` without additional arguments runs a default suite:
- **Skinny-small** (`64×2` · `2×10`) highlights the classic skinny case.
- **Skinny-large** (`20000×2` · `2×10`) stresses tall-and-skinny batches.
- **Square-small** (`128×128` · `128×128`) showcases medium square workloads.
- **Square-large** (`512×512` · `512×512`) samples a larger square GEMM.
- Additional experimental shapes (e.g. `20000×10` · `10×5`) probe corner cases
  where the cost model may prefer the naive loop.

Arguments follow `NAME=value` pairs:

- `M`, `K`, `N`: dimensions of matrices `A[M,K]`, `B[K,N]`.
- `I`: number of iterations to average (defaults to 10).
- `R`: optional number of random test cases to sample (prints aggregate
  accuracy for the cost model and a few mismatches).

The benchmark output lists each implementation with its average runtime in
milliseconds, the speedup factor relative to the naive kernel, and both the
cost-model prediction and the measured winner. You can vary the dimensions,
including non-divisible sizes (e.g. `M=123 K=245 N=67`), to gauge behavior
across workloads.

Under the hood, the cost model balances floating-point work versus memory
traffic to pick between the naive, tiled, and skinny (`K==2`) kernels. Use the
random sampling option (`R=<count>`) to sanity check the heuristic against real
measurements on your machine.
