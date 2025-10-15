#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

// Minimal Tensor API with separate, contiguous storage for data and grad.
// - ParameterStore owns two SoA buffers (data_buf, grad_buf).
// - Tensor is a lightweight view: offset + shape + numel into those buffers.

struct ParameterStore;

struct ParameterStoreStats {
  size_t tensor_zero_calls = 0;
  size_t tensor_zero_elems = 0;
  double tensor_zero_ms = 0.0;
  size_t zero_grad_calls = 0;
  size_t zero_grad_elems = 0;
  double zero_grad_ms = 0.0;
  size_t reserve_calls = 0;
  size_t reserve_elements = 0;
  size_t capacity_grow_events = 0;
  size_t peak_elements = 0;
};

enum class TensorInit {
  ZeroData,
  UninitializedData,
};

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
  std::unique_ptr<float[]> data_buf;
  std::unique_ptr<float[]> grad_buf;
  size_t capacity = 0;
  size_t used = 0;
  std::vector<TapeOp> tape;
  ParameterStoreStats stats;
  bool stats_enabled = false;
  size_t param_grad_offset = 0;
  size_t param_grad_span = 0;
  size_t param_grad_elements = 0;
  bool param_block_initialized = false;
  bool param_block_contiguous = true;

  // Reserve space for a tensor; returns starting offset.
  size_t allocate(size_t count);

  // Pre-allocate capacity for upcoming tensors (in elements).
  void reserve(size_t total_elements);
  void ensure_capacity(size_t required);

  // Memory scope helpers for reusing temporary tensors between iterations.
  size_t mark() const;
  void reset(size_t mark);

  // Introspection helpers
  size_t size() const { return used; }
  size_t capacity_count() const { return capacity; }
  float *data_ptr(size_t offset);
  const float *data_ptr(size_t offset) const;
  float *grad_ptr(size_t offset);
  const float *grad_ptr(size_t offset) const;

  // Factory helpers
  Tensor tensor(const std::vector<int> &shape,
                TensorInit init = TensorInit::UninitializedData);
  Tensor parameter(const std::vector<int> &shape, float scale = 0.01f,
                   unsigned seed = 0);

  // Stats controls
  void enable_stats(bool enabled = true);
  void reset_stats();
  const ParameterStoreStats &get_stats() const;
  bool stats_active() const { return stats_enabled; }
  void print_stats() const;

  // Bulk zero grads (optional convenience)
  void zero_grad();

  // Autograd controls
  void clear_tape();
  void backward(const Tensor &loss);

 private:
  void register_parameter_allocation(size_t offset, size_t count);
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
