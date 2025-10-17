# Transformer Library

A C++ library for machine learning experiments, focusing on tensor operations, neural networks, and autograd.

## Quick Start

1. Clone the repository.
2. Build: `./build.sh`
3. Run default experiment: `./run.sh`
4. List available commands: `./build/release/tformer --list`
5. Run specific experiment: `./run.sh <command>` (e.g., `bigram`, `mnist`)

For debug builds: `./build_debug.sh` and `./run_debug.sh`

## Guidelines

- Use std library and auto where possible.
- Run `./format.sh --check` before committing (enforced by pre-commit hook).

## Project Layout

- `apps/`: CLI entrypoints.
- `include/`: Public headers.
- `lib/`: Implementation by concern (core, nn, data, models, train, utils).

## Testing

Run tests: `./test.sh`

## Benchmarks

Microbenchmarks under `microbenchmarks/`. Run via `./bench.sh <name> [args]`.

Example: `./bench.sh matmul M=256 K=256 N=256`

See `bench.sh` for details.

## API Usage Examples

### Basic Tensor Operations

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

### Neural Network Modules

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

### Optimization

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

## Resources

- Dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Video: https://www.youtube.com/watch?v=kCc8FmEb1nY

For design details, see documentation or source code.
