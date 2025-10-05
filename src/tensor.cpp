#include "tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__clang__) || defined(__GNUC__)
#define TF_RESTRICT __restrict__
#else
#define TF_RESTRICT
#endif

#if defined(__clang__)
#define TF_PRAGMA(x) _Pragma(#x)
#define TF_VECTORIZE_LOOP TF_PRAGMA(clang loop vectorize(enable)) TF_PRAGMA(clang loop interleave(enable))
#define TF_VECTORIZE_REDUCTION(var) TF_VECTORIZE_LOOP
#else
#define TF_VECTORIZE_LOOP
#define TF_VECTORIZE_REDUCTION(var)
#endif

namespace {
size_t compute_numel(const std::vector<int> &shape) {
  size_t n = 1;
  TF_VECTORIZE_LOOP
  for (int d : shape) {
    if (d <= 0) throw std::invalid_argument("Tensor shape must be positive");
    n *= static_cast<size_t>(d);
  }
  return n;
}

inline void zero_buffer(float *ptr, size_t count) {
  if (!ptr || count == 0) return;
  std::memset(ptr, 0, count * sizeof(float));
}
}  // namespace

// Tensor methods
float *Tensor::data() {
  return store ? store->data_ptr(offset) : nullptr;
}
float *Tensor::grad() {
  return store ? store->grad_ptr(offset) : nullptr;
}
const float *Tensor::data() const {
  return store ? store->data_ptr(offset) : nullptr;
}
const float *Tensor::grad() const {
  return store ? store->grad_ptr(offset) : nullptr;
}

void Tensor::zero_grad() {
  if (!store) return;
  float *ptr = store->grad_ptr(offset);
  if (!ptr) return;
  zero_buffer(ptr, numel);
}

void Tensor::fill(float v) {
  if (!store) return;
  float *ptr = store->data_ptr(offset);
  if (!ptr) return;
  std::fill(ptr, ptr + numel, v);
}

// ParameterStore
size_t ParameterStore::allocate(size_t count) {
  const size_t off = used;
  if (count == 0) {
    if (stats_enabled) {
      stats.peak_elements = std::max(stats.peak_elements, used);
    }
    return off;
  }

  const size_t required = used + count;
  ensure_capacity(required);
  used = required;
  if (stats_enabled) {
    stats.peak_elements = std::max(stats.peak_elements, used);
  }
  return off;
}

void ParameterStore::reserve(size_t total_elements) {
  ensure_capacity(total_elements);
  if (stats_enabled) {
    stats.reserve_calls += 1;
    stats.reserve_elements =
        std::max(stats.reserve_elements, total_elements);
  }
}

void ParameterStore::ensure_capacity(size_t required) {
  if (required <= capacity) return;

  size_t new_capacity = capacity == 0 ? required : capacity;
  while (new_capacity < required) {
    new_capacity = std::max(new_capacity * 2, new_capacity + static_cast<size_t>(1024));
  }

  std::unique_ptr<float[]> new_data(new float[new_capacity]);
  std::unique_ptr<float[]> new_grad(new float[new_capacity]);

  if (data_buf) std::copy_n(data_buf.get(), used, new_data.get());
  if (grad_buf) std::copy_n(grad_buf.get(), used, new_grad.get());

  data_buf.swap(new_data);
  grad_buf.swap(new_grad);
  capacity = new_capacity;

  if (stats_enabled) {
    stats.capacity_grow_events += 1;
  }
}

float *ParameterStore::data_ptr(size_t offset) {
  return data_buf ? data_buf.get() + offset : nullptr;
}

const float *ParameterStore::data_ptr(size_t offset) const {
  return data_buf ? data_buf.get() + offset : nullptr;
}

float *ParameterStore::grad_ptr(size_t offset) {
  return grad_buf ? grad_buf.get() + offset : nullptr;
}

const float *ParameterStore::grad_ptr(size_t offset) const {
  return grad_buf ? grad_buf.get() + offset : nullptr;
}

Tensor ParameterStore::tensor(const std::vector<int> &shape, TensorInit init) {
  const bool zero_data = (init == TensorInit::ZeroData);
  const size_t n = compute_numel(shape);
  const size_t off = allocate(n);

  if (n == 0) return Tensor{this, off, shape, n};

  float *data = data_ptr(off);
  float *grad = grad_ptr(off);

  if (stats_enabled) {
    auto start = std::chrono::steady_clock::now();
    if (zero_data && data) {
      zero_buffer(data, n);
    }
    if (grad) {
      zero_buffer(grad, n);
    }
    auto end = std::chrono::steady_clock::now();
    stats.tensor_zero_calls += 1;
    stats.tensor_zero_elems += n + (zero_data ? n : 0);
    stats.tensor_zero_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
  } else {
    if (zero_data && data) {
      zero_buffer(data, n);
    }
    if (grad) {
      zero_buffer(grad, n);
    }
  }

  return Tensor{this, off, shape, n};
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
  float *grad_base = grad_ptr(0);
  const size_t count = used;
  if (!grad_base || count == 0) {
    if (stats_enabled) {
      stats.zero_grad_calls += 1;
    }
    return;
  }

  if (stats_enabled) {
    auto start = std::chrono::steady_clock::now();
    zero_buffer(grad_base, count);
    auto end = std::chrono::steady_clock::now();
    stats.zero_grad_calls += 1;
    stats.zero_grad_elems += count;
    stats.zero_grad_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
  } else {
    zero_buffer(grad_base, count);
  }
}

void ParameterStore::clear_tape() { tape.clear(); }

void ParameterStore::backward(const Tensor &loss) {
  if (loss.store != this)
    throw std::invalid_argument("loss belongs to different store");
  // Seed dL/dL = 1
  float *g = loss.store->grad_ptr(loss.offset);
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
        const float *TF_RESTRICT g_out = op.out.grad();
        const float *TF_RESTRICT x = op.a.data();
        float *TF_RESTRICT gx = op.a.grad();
        TF_VECTORIZE_LOOP
        for (size_t i = 0; i < op.out.numel; ++i) {
          gx[i] += g_out[i] * (x[i] > 0.0f ? 1.0f : 0.0f);
        }
        break;
      }
      case OpType::Tanh: {
        const float *TF_RESTRICT g_out = op.out.grad();
        const float *TF_RESTRICT y = op.out.data();
        float *TF_RESTRICT gx = op.a.grad();
        TF_VECTORIZE_LOOP
        for (size_t i = 0; i < op.out.numel; ++i) {
          gx[i] += g_out[i] * (1.0f - y[i] * y[i]);
        }
        break;
      }
      case OpType::Sigmoid: {
        // y = sigmoid(x) stored in out; dy/dx = y*(1-y)
        const float *TF_RESTRICT g_out = op.out.grad();
        const float *TF_RESTRICT y = op.out.data();
        float *TF_RESTRICT gx = op.a.grad();
        TF_VECTORIZE_LOOP
        for (size_t i = 0; i < op.out.numel; ++i) {
          gx[i] += g_out[i] * y[i] * (1.0f - y[i]);
        }
        break;
      }
      case OpType::Log: {
        // y = log(x); dy/dx = 1/x
        const float *TF_RESTRICT g_out = op.out.grad();
        const float *TF_RESTRICT x = op.a.data();
        float *TF_RESTRICT gx = op.a.grad();
        TF_VECTORIZE_LOOP
        for (size_t i = 0; i < op.out.numel; ++i) gx[i] += g_out[i] / x[i];
        break;
      }
      case OpType::Sum: {
        // out is scalar; upstream grad is shared across all elems
        const float g_out = op.out.grad()[0];
        float *TF_RESTRICT gx = op.a.grad();
        TF_VECTORIZE_LOOP
        for (size_t i = 0; i < op.a.numel; ++i) gx[i] += g_out;
        break;
      }
      case OpType::Matmul: {
        // Y = A[M,K] x B[K,N]
        int M = op.a.shape[0];
        int K = op.a.shape[1];
        int N = op.b.shape[1];
        const float *TF_RESTRICT A = op.a.data();
        const float *TF_RESTRICT B = op.b.data();
        const float *TF_RESTRICT gY = op.out.grad();
        float *TF_RESTRICT gA = op.a.grad();
        float *TF_RESTRICT gB = op.b.grad();
        // dA = gY x B^T
        for (int m = 0; m < M; ++m) {
          for (int k = 0; k < K; ++k) {
            float acc = 0.0f;
            TF_VECTORIZE_REDUCTION(acc)
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
            TF_VECTORIZE_REDUCTION(acc)
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
        float *TF_RESTRICT gX = op.a.grad();
        float *TF_RESTRICT gb = op.b.grad();
        // dX = g_out
        TF_VECTORIZE_LOOP
        for (int i = 0; i < N * H; ++i) gX[i] += g_out[i];
        // db[h] = sum_i g_out[i,h]
        for (int h = 0; h < H; ++h) {
          float acc = 0.0f;
          TF_VECTORIZE_REDUCTION(acc)
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
  const float *TF_RESTRICT ap = a.data();
  const float *TF_RESTRICT bp = b.data();
  float *TF_RESTRICT op = out.data();
  TF_VECTORIZE_LOOP
  for (size_t i = 0; i < a.numel; ++i) op[i] = ap[i] + bp[i];
  store.tape.push_back(TapeOp{OpType::Add, out, a, b});
  return out;
}

Tensor sub(const Tensor &a, const Tensor &b, ParameterStore &store) {
  assert_same_shape(a, b);
  Tensor out = store.tensor(a.shape);
  const float *TF_RESTRICT ap = a.data();
  const float *TF_RESTRICT bp = b.data();
  float *TF_RESTRICT op = out.data();
  TF_VECTORIZE_LOOP
  for (size_t i = 0; i < a.numel; ++i) op[i] = ap[i] - bp[i];
  store.tape.push_back(TapeOp{OpType::Sub, out, a, b});
  return out;
}

Tensor mul(const Tensor &a, const Tensor &b, ParameterStore &store) {
  assert_same_shape(a, b);
  Tensor out = store.tensor(a.shape);
  const float *TF_RESTRICT ap = a.data();
  const float *TF_RESTRICT bp = b.data();
  float *TF_RESTRICT op = out.data();
  TF_VECTORIZE_LOOP
  for (size_t i = 0; i < a.numel; ++i) op[i] = ap[i] * bp[i];
  store.tape.push_back(TapeOp{OpType::Mul, out, a, b});
  return out;
}

Tensor relu(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor(x.shape);
  const float *TF_RESTRICT xp = x.data();
  float *TF_RESTRICT op = out.data();
  TF_VECTORIZE_LOOP
  for (size_t i = 0; i < x.numel; ++i) op[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
  store.tape.push_back(TapeOp{OpType::Relu, out, x, Tensor{}});
  return out;
}

Tensor vtanh(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor(x.shape);
  const float *TF_RESTRICT xp = x.data();
  float *TF_RESTRICT op = out.data();
  TF_VECTORIZE_LOOP
  for (size_t i = 0; i < x.numel; ++i) op[i] = std::tanh(xp[i]);
  store.tape.push_back(TapeOp{OpType::Tanh, out, x, Tensor{}});
  return out;
}

Tensor sigmoid(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor(x.shape);
  const float *TF_RESTRICT xp = x.data();
  float *TF_RESTRICT op = out.data();
  TF_VECTORIZE_LOOP
  for (size_t i = 0; i < x.numel; ++i) op[i] = 1.0f / (1.0f + std::exp(-xp[i]));
  store.tape.push_back(TapeOp{OpType::Sigmoid, out, x, Tensor{}});
  return out;
}

Tensor vlog(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor(x.shape);
  const float *TF_RESTRICT xp = x.data();
  float *TF_RESTRICT op = out.data();
  TF_VECTORIZE_LOOP
  for (size_t i = 0; i < x.numel; ++i) op[i] = std::log(xp[i]);
  store.tape.push_back(TapeOp{OpType::Log, out, x, Tensor{}});
  return out;
}

Tensor sum(const Tensor &x, ParameterStore &store) {
  Tensor out = store.tensor({1});
  float acc = 0.0f;
  const float *TF_RESTRICT xp = x.data();
  TF_VECTORIZE_REDUCTION(acc)
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

  const float *TF_RESTRICT A = a.data();
  const float *TF_RESTRICT B = b.data();
  float *TF_RESTRICT C = out.data();

  zero_buffer(C, static_cast<size_t>(M) * static_cast<size_t>(N));

  for (int m = 0; m < M; ++m) {
    const float *TF_RESTRICT Arow = A + m * K;
    float *TF_RESTRICT Crow = C + m * N;
    for (int k = 0; k < K; ++k) {
      const float a_val = Arow[k];
      const float *TF_RESTRICT Brow = B + k * N;
      TF_VECTORIZE_LOOP
      for (int n = 0; n < N; ++n) {
        Crow[n] += a_val * Brow[n];
      }
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
  const float *TF_RESTRICT xp = X.data();
  const float *TF_RESTRICT bp = b.data();
  float *TF_RESTRICT op = out.data();
  for (int n = 0; n < N; ++n) {
    const float *TF_RESTRICT xrow = xp + n * H;
    float *TF_RESTRICT orow = op + n * H;
    TF_VECTORIZE_LOOP
    for (int h = 0; h < H; ++h) {
      orow[h] = xrow[h] + bp[h];
    }
  }
  store.tape.push_back(TapeOp{OpType::AddRowwise, out, X, b});
  return out;
}
