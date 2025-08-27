#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

// Minimal Tensor API with separate, contiguous storage for data and grad.
// - ParameterStore owns two SoA buffers (data_buf, grad_buf).
// - Tensor is a lightweight view: offset + shape + numel into those buffers.

struct ParameterStore;

enum class OpType {
  Add,
  Sub,
  Mul,
  Relu,
  Tanh,
  Sigmoid,
  Log,
  Sum,
  Matmul,
  AddRowwise
};

struct Tensor {
  ParameterStore *store = nullptr;
  size_t offset = 0;       // start index into store buffers
  std::vector<int> shape;  // simple, contiguous layout
  size_t numel = 0;        // product(shape)

  Tensor() = default;
  Tensor(ParameterStore *s, size_t off, std::vector<int> sh, size_t n)
      : store(s), offset(off), shape(std::move(sh)), numel(n) {}

  float *data();
  float *grad();
  const float *data() const;
  const float *grad() const;

  void zero_grad();
  void fill(float v);
};

struct TapeOp {
  OpType type;
  Tensor out;
  // For binary ops (Add, Mul, Matmul) use a and b; for unary (Relu, Sum) use a
  Tensor a;
  Tensor b;  // unused for unary
};

struct ParameterStore {
  std::vector<float> data_buf;
  std::vector<float> grad_buf;
  std::vector<TapeOp> tape;

  // Reserve space for a tensor; returns starting offset.
  size_t allocate(size_t count);

  // Factory helpers
  Tensor tensor(const std::vector<int> &shape);
  Tensor parameter(const std::vector<int> &shape, float scale = 0.01f,
                   unsigned seed = 0);

  // Bulk zero grads (optional convenience)
  void zero_grad();

  // Autograd controls
  void clear_tape();
  void backward(const Tensor &loss);
};

// Basic elementwise ops (contiguous, same-shape only; minimal checks)
Tensor add(const Tensor &a, const Tensor &b, ParameterStore &store);
Tensor sub(const Tensor &a, const Tensor &b, ParameterStore &store);
Tensor mul(const Tensor &a, const Tensor &b, ParameterStore &store);

// Simple matrix multiply: a[M,K] x b[K,N] -> out[M,N]
Tensor matmul(const Tensor &a, const Tensor &b, ParameterStore &store);

// Unary ops
Tensor relu(const Tensor &x, ParameterStore &store);
Tensor vtanh(const Tensor &x, ParameterStore &store);
Tensor sigmoid(const Tensor &x, ParameterStore &store);
Tensor vlog(const Tensor &x, ParameterStore &store);  // natural log

// Reductions
Tensor sum(const Tensor &x, ParameterStore &store);  // returns scalar [1]

// Add bias vector b[H] to each row of X[N,H]
Tensor add_rowwise(const Tensor &X, const Tensor &b, ParameterStore &store);
