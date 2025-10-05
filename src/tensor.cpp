#include "tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
size_t compute_numel(const std::vector<int> &shape) {
  size_t n = 1;
  for (int d : shape) {
    if (d <= 0) throw std::invalid_argument("Tensor shape must be positive");
    n *= static_cast<size_t>(d);
  }
  return n;
}
}  // namespace

// Tensor methods
float *Tensor::data() {
  return store ? store->data_buf.data() + offset : nullptr;
}
float *Tensor::grad() {
  return store ? store->grad_buf.data() + offset : nullptr;
}
const float *Tensor::data() const {
  return store ? store->data_buf.data() + offset : nullptr;
}
const float *Tensor::grad() const {
  return store ? store->grad_buf.data() + offset : nullptr;
}

void Tensor::zero_grad() {
  if (!store) return;
  std::fill(store->grad_buf.begin() + offset,
            store->grad_buf.begin() + offset + numel, 0.0f);
}

void Tensor::fill(float v) {
  if (!store) return;
  std::fill(store->data_buf.begin() + offset,
            store->data_buf.begin() + offset + numel, v);
}

// ParameterStore
size_t ParameterStore::allocate(size_t count) {
  size_t off = data_buf.size();
  data_buf.resize(off + count);
  grad_buf.resize(off + count);
  return off;
}

Tensor ParameterStore::tensor(const std::vector<int> &shape) {
  if (stats_enabled) {
    auto start = std::chrono::steady_clock::now();
    auto n = compute_numel(shape);
    auto off = allocate(n);
    std::fill(data_buf.begin() + off, data_buf.begin() + off + n, 0.0f);
    std::fill(grad_buf.begin() + off, grad_buf.begin() + off + n, 0.0f);
    auto end = std::chrono::steady_clock::now();
    stats.tensor_zero_calls += 1;
    stats.tensor_zero_elems += n * 2;
    stats.tensor_zero_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
    return Tensor{this, off, shape, n};
  } else {
    auto n = compute_numel(shape);
    auto off = allocate(n);
    std::fill(data_buf.begin() + off, data_buf.begin() + off + n, 0.0f);
    std::fill(grad_buf.begin() + off, grad_buf.begin() + off + n, 0.0f);
    return Tensor{this, off, shape, n};
  }
}

Tensor ParameterStore::parameter(const std::vector<int> &shape, float scale,
                                 unsigned seed) {
  auto t = tensor(shape);
  std::mt19937 gen(seed ? seed : std::random_device{}());
  std::uniform_real_distribution<float> dist(-scale, scale);
  auto *p = t.data();
  for (size_t i = 0; i < t.numel; ++i) p[i] = dist(gen);
  return t;
}

void ParameterStore::enable_stats(bool enabled) {
  stats_enabled = enabled;
  reset_stats();
}

void ParameterStore::reset_stats() { stats = ParameterStoreStats{}; }

const ParameterStoreStats &ParameterStore::get_stats() const { return stats; }

void ParameterStore::zero_grad() {
  if (stats_enabled) {
    auto start = std::chrono::steady_clock::now();
    std::fill(grad_buf.begin(), grad_buf.end(), 0.0f);
    auto end = std::chrono::steady_clock::now();
    stats.zero_grad_calls += 1;
    stats.zero_grad_elems += grad_buf.size();
    stats.zero_grad_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
  } else {
    std::fill(grad_buf.begin(), grad_buf.end(), 0.0f);
  }
}

void ParameterStore::clear_tape() { tape.clear(); }

void ParameterStore::backward(const Tensor &loss) {
  if (loss.store != this)
    throw std::invalid_argument("loss belongs to different store");
  // Seed dL/dL = 1
  float *g = loss.store->grad_buf.data() + loss.offset;
  if (loss.numel == 1) {
    g[0] += 1.0f;
  } else {
    for (size_t i = 0; i < loss.numel; ++i) g[i] += 1.0f;
  }
  // Traverse tape in reverse
  for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
    TapeOp &op = *it;
    switch (op.type) {
      case OpType::Add: {
        const float *g_out = op.out.grad();
        float *ga = op.a.grad();
        float *gb = op.b.grad();
        for (size_t i = 0; i < op.out.numel; ++i) {
          ga[i] += g_out[i];
          gb[i] += g_out[i];
        }
        break;
      }
      case OpType::Sub: {
        const float *g_out = op.out.grad();
        float *ga = op.a.grad();
        float *gb = op.b.grad();
        for (size_t i = 0; i < op.out.numel; ++i) {
          ga[i] += g_out[i];
          gb[i] -= g_out[i];
        }
        break;
      }
      case OpType::Mul: {
        const float *g_out = op.out.grad();
        const float *a_data = op.a.data();
        const float *b_data = op.b.data();
        float *ga = op.a.grad();
        float *gb = op.b.grad();
        for (size_t i = 0; i < op.out.numel; ++i) {
          ga[i] += g_out[i] * b_data[i];
          gb[i] += g_out[i] * a_data[i];
        }
        break;
      }
      case OpType::Relu: {
        const float *g_out = op.out.grad();
        const float *x = op.a.data();
        float *gx = op.a.grad();
        for (size_t i = 0; i < op.out.numel; ++i) {
          gx[i] += g_out[i] * (x[i] > 0.0f ? 1.0f : 0.0f);
        }
        break;
      }
      case OpType::Tanh: {
        const float *g_out = op.out.grad();
        const float *y = op.out.data();
        float *gx = op.a.grad();
        for (size_t i = 0; i < op.out.numel; ++i) {
          gx[i] += g_out[i] * (1.0f - y[i] * y[i]);
        }
        break;
      }
      case OpType::Sigmoid: {
        // y = sigmoid(x) stored in out; dy/dx = y*(1-y)
        const float *g_out = op.out.grad();
        const float *y = op.out.data();
        float *gx = op.a.grad();
        for (size_t i = 0; i < op.out.numel; ++i) {
          gx[i] += g_out[i] * y[i] * (1.0f - y[i]);
        }
        break;
      }
      case OpType::Log: {
        // y = log(x); dy/dx = 1/x
        const float *g_out = op.out.grad();
        const float *x = op.a.data();
        float *gx = op.a.grad();
        for (size_t i = 0; i < op.out.numel; ++i) gx[i] += g_out[i] / x[i];
        break;
      }
      case OpType::Sum: {
        // out is scalar; upstream grad is shared across all elems
        const float g_out = op.out.grad()[0];
        float *gx = op.a.grad();
        for (size_t i = 0; i < op.a.numel; ++i) gx[i] += g_out;
        break;
      }
      case OpType::Matmul: {
        // Y = A[M,K] x B[K,N]
        int M = op.a.shape[0];
        int K = op.a.shape[1];
        int N = op.b.shape[1];
        const float *A = op.a.data();
        const float *B = op.b.data();
        const float *gY = op.out.grad();
        float *gA = op.a.grad();
        float *gB = op.b.grad();
        // dA = gY x B^T
        for (int m = 0; m < M; ++m) {
          for (int k = 0; k < K; ++k) {
            float acc = 0.0f;
            for (int n = 0; n < N; ++n) {
              acc += gY[m * N + n] * B[k * N + n];
            }
            gA[m * K + k] += acc;
          }
        }
        // dB = A^T x gY
        for (int k = 0; k < K; ++k) {
          for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int m = 0; m < M; ++m) {
              acc += A[m * K + k] * gY[m * N + n];
            }
            gB[k * N + n] += acc;
          }
        }
        break;
      }
      case OpType::AddRowwise: {
        // X[N,H] + b[H]
        int N = op.a.shape[0];
        int H = op.a.shape[1];
        const float *g_out = op.out.grad();
        float *gX = op.a.grad();
        float *gb = op.b.grad();
        // dX = g_out
        for (int i = 0; i < N * H; ++i) gX[i] += g_out[i];
        // db[h] = sum_i g_out[i,h]
        for (int h = 0; h < H; ++h) {
          float acc = 0.0f;
          for (int n = 0; n < N; ++n) acc += g_out[n * H + h];
          gb[h] += acc;
        }
        break;
      }
    }
  }
}

// Ops
static void assert_same_shape(const Tensor &a, const Tensor &b) {
  if (a.shape.size() != b.shape.size())
    throw std::invalid_argument("Shape rank mismatch");
  if (a.numel != b.numel) throw std::invalid_argument("Numel mismatch");
  for (size_t i = 0; i < a.shape.size(); ++i)
    if (a.shape[i] != b.shape[i]) throw std::invalid_argument("Shape mismatch");
}

Tensor add(const Tensor &a, const Tensor &b, ParameterStore &store) {
  assert_same_shape(a, b);
  Tensor out = store.tensor(a.shape);
  const float *ap = a.data();
  const float *bp = b.data();
  float *op = out.data();
  for (size_t i = 0; i < a.numel; ++i) op[i] = ap[i] + bp[i];
  store.tape.push_back(TapeOp{OpType::Add, out, a, b});
  return out;
}

Tensor sub(const Tensor &a, const Tensor &b, ParameterStore &store) {
  assert_same_shape(a, b);
  Tensor out = store.tensor(a.shape);
  const float *ap = a.data();
  const float *bp = b.data();
  float *op = out.data();
  for (size_t i = 0; i < a.numel; ++i) op[i] = ap[i] - bp[i];
  store.tape.push_back(TapeOp{OpType::Sub, out, a, b});
  return out;
}

Tensor mul(const Tensor &a, const Tensor &b, ParameterStore &store) {
  assert_same_shape(a, b);
  Tensor out = store.tensor(a.shape);
  const float *ap = a.data();
  const float *bp = b.data();
  float *op = out.data();
  for (size_t i = 0; i < a.numel; ++i) op[i] = ap[i] * bp[i];
  store.tape.push_back(TapeOp{OpType::Mul, out, a, b});
  return out;
}

Tensor relu(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor(x.shape);
  const float *xp = x.data();
  float *op = out.data();
  for (size_t i = 0; i < x.numel; ++i) op[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
  store.tape.push_back(TapeOp{OpType::Relu, out, x, Tensor{}});
  return out;
}

Tensor vtanh(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor(x.shape);
  const float *xp = x.data();
  float *op = out.data();
  for (size_t i = 0; i < x.numel; ++i) op[i] = std::tanh(xp[i]);
  store.tape.push_back(TapeOp{OpType::Tanh, out, x, Tensor{}});
  return out;
}

Tensor sigmoid(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor(x.shape);
  const float *xp = x.data();
  float *op = out.data();
  for (size_t i = 0; i < x.numel; ++i) op[i] = 1.0f / (1.0f + std::exp(-xp[i]));
  store.tape.push_back(TapeOp{OpType::Sigmoid, out, x, Tensor{}});
  return out;
}

Tensor vlog(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor(x.shape);
  const float *xp = x.data();
  float *op = out.data();
  for (size_t i = 0; i < x.numel; ++i) op[i] = std::log(xp[i]);
  store.tape.push_back(TapeOp{OpType::Log, out, x, Tensor{}});
  return out;
}

Tensor sum(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor({1});
  float acc = 0.0f;
  const float *xp = x.data();
  for (size_t i = 0; i < x.numel; ++i) acc += xp[i];
  out.data()[0] = acc;
  store.tape.push_back(TapeOp{OpType::Sum, out, x, Tensor{}});
  return out;
}

Tensor matmul(const Tensor &a, const Tensor &b, ParameterStore &store) {
  if (a.shape.size() != 2 || b.shape.size() != 2)
    throw std::invalid_argument("matmul expects 2D tensors");
  int M = a.shape[0];
  int K = a.shape[1];
  int K2 = b.shape[0];
  int N = b.shape[1];
  if (K != K2) throw std::invalid_argument("matmul inner dim mismatch");
  Tensor out = store.tensor({M, N});

  const float *A = a.data();
  const float *B = b.data();
  float *C = out.data();
  // naive triple loop (row-major contiguous)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = acc;
    }
  }
  store.tape.push_back(TapeOp{OpType::Matmul, out, a, b});
  return out;
}

Tensor add_rowwise(const Tensor &X, const Tensor &b, ParameterStore &store) {
  if (X.shape.size() != 2 || b.shape.size() != 1)
    throw std::invalid_argument("add_rowwise expects X[N,H], b[H]");
  int N = X.shape[0];
  int H = X.shape[1];
  if (b.shape[0] != H) throw std::invalid_argument("add_rowwise dim mismatch");
  Tensor out = store.tensor({N, H});
  const float *xp = X.data();
  const float *bp = b.data();
  float *op = out.data();
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      op[n * H + h] = xp[n * H + h] + bp[h];
    }
  }
  store.tape.push_back(TapeOp{OpType::AddRowwise, out, X, b});
  return out;
}
