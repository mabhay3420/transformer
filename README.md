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

# API Usage Examples

## Basic Tensor Operations

```cpp
#include "tensor.hpp"

// Create a parameter store for memory management
ParameterStore store;

// Create tensors
auto a = store.tensor({2, 3});  // 2x3 uninitialized tensor
auto b = store.parameter({3, 4}, 0.1f);  // 3x4 parameter with scale 0.1

// Perform operations
auto c = matmul(a, b, store);  // Matrix multiplication
auto d = relu(c, store);       // ReLU activation
auto loss = sum(d, store);     // Sum to scalar

// Compute gradients
store.backward(loss);
```

## Neural Network Modules

```cpp
#include "nn.hpp"
#include "tensor.hpp"

ParameterStore store;

// Create a simple MLP
nn::Sequential model;
model.emplace_back<nn::Linear>(784, 128, store);  // Input to hidden
model.emplace_back<nn::Relu>();                    // Activation
model.emplace_back<nn::Linear>(128, 10, store);   // Hidden to output

// Forward pass
auto input = store.tensor({32, 784});  // Batch of 32 samples
auto logits = model.forward(input, store);

// Compute loss
auto targets = store.tensor({32, 10});  // One-hot targets
auto loss = nn::bce_with_logits_loss(logits, targets, store);
```

## Optimization

```cpp
#include "optimizer.hpp"
#include "learning_rate.hpp"

// Get model parameters
auto params = model.params();

// Create optimizer with scheduler
ConstantLRScheduler scheduler(0.01f);
optim::AdamW optimizer(params, scheduler, 0.9f, 0.999f, 0.01f);

// Training loop
for (int epoch = 0; epoch < 100; ++epoch) {
    // Forward pass...
    // Compute loss...

    store.zero_grad();
    store.backward(loss);
    optimizer.step();
}
```

# Design Decisions and Trade-offs

## Memory Management
- **ParameterStore**: Centralizes memory allocation to minimize fragmentation and enable efficient reuse.
- **Tape-based Autograd**: Records operations for reverse-mode differentiation, trading memory for flexibility.
- **Contiguous Buffers**: SoA layout (separate data/grad) optimizes cache locality and SIMD operations.

## Performance Considerations
- **MLX Integration**: Leverages Apple's MLX library for optimized matrix operations on CPU/GPU.
- **SIMD Optimizations**: NEON intrinsics for ARM processors, with fallback implementations.
- **Memory Reuse**: Mark/reset pattern allows temporary tensor reuse between iterations.

## API Design
- **RAII**: Smart pointers and destructors handle resource cleanup automatically.
- **CRTP Schedulers**: Compile-time polymorphism for zero-overhead learning rate scheduling.
- **Template Optimizers**: Type-safe optimizer variants with scheduler integration.

## Trade-offs
- **Memory vs Speed**: Tape recording increases memory usage but enables arbitrary computation graphs.
- **Generality vs Optimization**: Unified tensor operations support various workloads at cost of specialization.
- **C++ Standards**: Uses modern C++17 features for expressiveness, requiring recent compilers.
