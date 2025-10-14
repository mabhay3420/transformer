#include "tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "matmul_cost_model.hpp"

namespace {
#if defined(__clang__) || defined(__GNUC__)
#define UNROLL4 _Pragma("unroll 4")
#else
#define UNROLL4
#endif
size_t compute_numel(const std::vector<int> &shape) {
  size_t n = 1;
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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
inline float horizontal_add(float32x4_t v) {
  float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  sum = vpadd_f32(sum, sum);
  return vget_lane_f32(sum, 0);
}
#endif
}  // namespace

namespace {

void backward_add(TapeOp &op) {
  const float *g_out = op.out.grad();
  float *ga = op.a.grad();
  float *gb = op.b.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    ga[i] += g_out[i];
    gb[i] += g_out[i];
  }
}

void backward_sub(TapeOp &op) {
  const float *g_out = op.out.grad();
  float *ga = op.a.grad();
  float *gb = op.b.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    ga[i] += g_out[i];
    gb[i] -= g_out[i];
  }
}

void backward_mul(TapeOp &op) {
  const float *g_out = op.out.grad();
  const float *a_data = op.a.data();
  const float *b_data = op.b.data();
  float *ga = op.a.grad();
  float *gb = op.b.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    ga[i] += g_out[i] * b_data[i];
    gb[i] += g_out[i] * a_data[i];
  }
}

void backward_relu(TapeOp &op) {
  const float *g_out = op.out.grad();
  const float *x = op.a.data();
  float *gx = op.a.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    gx[i] += g_out[i] * (x[i] > 0.0f ? 1.0f : 0.0f);
  }
}

void backward_tanh(TapeOp &op) {
  const float *g_out = op.out.grad();
  const float *y = op.out.data();
  float *gx = op.a.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    gx[i] += g_out[i] * (1.0f - y[i] * y[i]);
  }
}

void backward_sigmoid(TapeOp &op) {
  const float *g_out = op.out.grad();
  const float *y = op.out.data();
  float *gx = op.a.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    gx[i] += g_out[i] * y[i] * (1.0f - y[i]);
  }
}

void backward_log(TapeOp &op) {
  const float *g_out = op.out.grad();
  const float *x = op.a.data();
  float *gx = op.a.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    gx[i] += g_out[i] / x[i];
  }
}

void backward_sum(TapeOp &op) {
  const float g_out = op.out.grad()[0];
  float *gx = op.a.grad();
  for (size_t i = 0; i < op.a.numel; ++i) {
    gx[i] += g_out;
  }
}

void backward_matmul_skinny(const float *A, const float *B, const float *gY,
                            float *gA, float *gB, int M, int N, int K) {
  const float *b0 = B;
  const float *b1 = B + N;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  for (int m = 0; m < M; ++m) {
    const float *gY_row = gY + m * N;
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int n = 0;
    for (; n + 4 <= N; n += 4) {
      const float32x4_t gy_vec = vld1q_f32(gY_row + n);
      const float32x4_t b0_vec = vld1q_f32(b0 + n);
      const float32x4_t b1_vec = vld1q_f32(b1 + n);
      sum0 = vmlaq_f32(sum0, gy_vec, b0_vec);
      sum1 = vmlaq_f32(sum1, gy_vec, b1_vec);
    }
    float acc0 = horizontal_add(sum0);
    float acc1 = horizontal_add(sum1);
    for (; n < N; ++n) {
      float gy_val = gY_row[n];
      acc0 += gy_val * b0[n];
      acc1 += gy_val * b1[n];
    }
    float *gA_row = gA + m * K;
    gA_row[0] += acc0;
    gA_row[1] += acc1;
  }
#else
  for (int m = 0; m < M; ++m) {
    const float *gY_row = gY + m * N;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    UNROLL4
    for (int n = 0; n < N; ++n) {
      float gy_val = gY_row[n];
      acc0 += gy_val * b0[n];
      acc1 += gy_val * b1[n];
    }
    float *gA_row = gA + m * K;
    gA_row[0] += acc0;
    gA_row[1] += acc1;
  }
#endif
  for (int n = 0; n < N; ++n) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    UNROLL4
    for (int m = 0; m < M; ++m) {
      const float *a_row = A + m * K;
      float gy_val = gY[m * N + n];
      acc0 += a_row[0] * gy_val;
      acc1 += a_row[1] * gy_val;
    }
    gB[n] += acc0;
    gB[N + n] += acc1;
  }
}

void backward_matmul_naive(const float *A, const float *B, const float *gY,
                           float *gA, float *gB, int M, int N, int K) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  for (int m = 0; m < M; ++m) {
    const float *gY_row = gY + m * N;
    for (int k = 0; k < K; ++k) {
      float32x4_t sum = vdupq_n_f32(0.0f);
      int n = 0;
      for (; n + 4 <= N; n += 4) {
        const float32x4_t gy_vec = vld1q_f32(gY_row + n);
        const float32x4_t b_vec = vld1q_f32(B + k * N + n);
        sum = vmlaq_f32(sum, gy_vec, b_vec);
      }
      float acc = horizontal_add(sum);
      for (; n < N; ++n) {
        acc += gY_row[n] * B[k * N + n];
      }
      gA[m * K + k] += acc;
    }
  }
#else
  for (int m = 0; m < M; ++m) {
    const float *gY_row = gY + m * N;
    for (int k = 0; k < K; ++k) {
      float acc = 0.0f;
      UNROLL4
      for (int n = 0; n < N; ++n) {
        acc += gY_row[n] * B[k * N + n];
      }
      gA[m * K + k] += acc;
    }
  }
#endif
  for (int k = 0; k < K; ++k) {
    float *gB_row = gB + k * N;
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      UNROLL4
      for (int m = 0; m < M; ++m) {
        acc += A[m * K + k] * gY[m * N + n];
      }
      gB_row[n] += acc;
    }
  }
}

template <int TILE_SIZE>
void backward_matmul_tiled(const float *A, const float *B, const float *gY,
                           float *gA, float *gB, int M, int N, int K) {
  for (int m0 = 0; m0 < M; m0 += TILE_SIZE) {
    int m_max = std::min(m0 + TILE_SIZE, M);
    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
      int k_block = std::min(k0 + TILE_SIZE, K) - k0;
      for (int m = m0; m < m_max; ++m) {
        float accum[TILE_SIZE];
        zero_buffer(accum, static_cast<size_t>(k_block));
        for (int n0 = 0; n0 < N; n0 += TILE_SIZE) {
          int n_max = std::min(n0 + TILE_SIZE, N);
          int n_block = n_max - n0;
          const float *gY_ptr = gY + m * N + n0;
          for (int ni = 0; ni < n_block; ++ni) {
            float gy_val = gY_ptr[ni];
            const float *b_ptr = B + k0 * N + (n0 + ni);
            for (int ki = 0; ki < k_block; ++ki) {
              accum[ki] += gy_val * b_ptr[ki * N];
            }
          }
        }
        float *gA_row = gA + m * K + k0;
        for (int ki = 0; ki < k_block; ++ki) {
          gA_row[ki] += accum[ki];
        }
      }
    }
  }

  for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
    int k_block = std::min(k0 + TILE_SIZE, K) - k0;
    for (int n0 = 0; n0 < N; n0 += TILE_SIZE) {
      int n_max = std::min(n0 + TILE_SIZE, N);
      int n_block = n_max - n0;
      for (int k = 0; k < k_block; ++k) {
        float accum[TILE_SIZE];
        zero_buffer(accum, static_cast<size_t>(n_block));
        for (int m0 = 0; m0 < M; m0 += TILE_SIZE) {
          int m_max = std::min(m0 + TILE_SIZE, M);
          for (int m = m0; m < m_max; ++m) {
            float a_val = A[m * K + (k0 + k)];
            const float *gY_ptr = gY + m * N + n0;
            for (int ni = 0; ni < n_block; ++ni) {
              accum[ni] += a_val * gY_ptr[ni];
            }
          }
        }
        float *gB_row = gB + (k0 + k) * N + n0;
        for (int ni = 0; ni < n_block; ++ni) {
          gB_row[ni] += accum[ni];
        }
      }
    }
  }
}

// TODO - First benchmark and then use M-Series `Accelerate` Matmul kernel
void backward_matmul(TapeOp &op) {
  int M = op.a.shape[0];
  int K = op.a.shape[1];
  int N = op.b.shape[1];
  const float *A = op.a.data();
  const float *B = op.b.data();
  const float *gY = op.out.grad();
  float *gA = op.a.grad();
  float *gB = op.b.grad();
  constexpr int TILE = 32;
  MatmulKernel kernel = predict_matmul_kernel(M, K, N);
  if (kernel == MatmulKernel::Skinny && K != 2) kernel = MatmulKernel::Naive;

  switch (kernel) {
    case MatmulKernel::Skinny:
      backward_matmul_skinny(A, B, gY, gA, gB, M, N, K);
      break;
    case MatmulKernel::Naive:
      backward_matmul_naive(A, B, gY, gA, gB, M, N, K);
      break;
    case MatmulKernel::Tiled:
      backward_matmul_tiled<TILE>(A, B, gY, gA, gB, M, N, K);
      break;
  }
}

void backward_add_rowwise(TapeOp &op) {
  int N = op.a.shape[0];
  int H = op.a.shape[1];
  const float *g_out = op.out.grad();
  float *gX = op.a.grad();
  float *gb = op.b.grad();
  for (int i = 0; i < N * H; ++i) {
    gX[i] += g_out[i];
  }
  for (int h = 0; h < H; ++h) {
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      acc += g_out[n * H + h];
    }
    gb[h] += acc;
  }
}

}  // namespace

// Tensor methods
float *Tensor::data() { return store ? store->data_ptr(offset) : nullptr; }
float *Tensor::grad() { return store ? store->grad_ptr(offset) : nullptr; }
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
    stats.reserve_elements = std::max(stats.reserve_elements, total_elements);
  }
}

void ParameterStore::ensure_capacity(size_t required) {
  if (required <= capacity) return;

  size_t new_capacity = capacity == 0 ? required : capacity;
  while (new_capacity < required) {
    new_capacity =
        std::max(new_capacity * 2, new_capacity + static_cast<size_t>(1024));
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

void ParameterStore::print_stats() const {
  using std::cout;
  using std::endl;
  const double tensor_avg_ms =
      stats.tensor_zero_calls
          ? stats.tensor_zero_ms / static_cast<double>(stats.tensor_zero_calls)
          : 0.0;
  const double zero_grad_avg_ms =
      stats.zero_grad_calls
          ? stats.zero_grad_ms / static_cast<double>(stats.zero_grad_calls)
          : 0.0;
  const double tensor_bytes =
      static_cast<double>(stats.tensor_zero_elems) * sizeof(float);
  const double zero_grad_bytes =
      static_cast<double>(stats.zero_grad_elems) * sizeof(float);
  const double tensor_mb = tensor_bytes / (1024.0 * 1024.0);
  const double zero_grad_mb = zero_grad_bytes / (1024.0 * 1024.0);

  cout << "ParameterStore zeroing stats:" << endl;
  cout << "  tensor() zero fills: " << stats.tensor_zero_calls
       << " calls, elements zeroed: " << stats.tensor_zero_elems
       << ", bytes zeroed: " << tensor_bytes << " (" << tensor_mb << " MB)"
       << ", total ms: " << stats.tensor_zero_ms
       << ", avg ms/call: " << tensor_avg_ms << endl;
  cout << "  zero_grad(): " << stats.zero_grad_calls
       << " calls, elements zeroed: " << stats.zero_grad_elems
       << ", bytes zeroed: " << zero_grad_bytes << " (" << zero_grad_mb
       << " MB)"
       << ", total ms: " << stats.zero_grad_ms
       << ", avg ms/call: " << zero_grad_avg_ms << endl;
  cout << "  reserve(): " << stats.reserve_calls
       << " calls, max hinted elements: " << stats.reserve_elements << endl;
  cout << "  capacity growth events: " << stats.capacity_grow_events
       << " (peak elements: " << stats.peak_elements << ")" << endl;
}

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
      case OpType::Add:
        backward_add(op);
        break;
      case OpType::Sub:
        backward_sub(op);
        break;
      case OpType::Mul:
        backward_mul(op);
        break;
      case OpType::Relu:
        backward_relu(op);
        break;
      case OpType::Tanh:
        backward_tanh(op);
        break;
      case OpType::Sigmoid:
        backward_sigmoid(op);
        break;
      case OpType::Log:
        backward_log(op);
        break;
      case OpType::Sum:
        backward_sum(op);
        break;
      case OpType::Matmul:
        backward_matmul(op);
        break;
      case OpType::AddRowwise:
        backward_add_rowwise(op);
        break;
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

void matmul_naive(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  for (int m = 0; m < M; ++m) {
    const float *a_row = A + m * K;
    float *c_row = C + m * N;
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      UNROLL4
      for (int k = 0; k < K; ++k) {
        acc += a_row[k] * B[k * N + n];
      }
      c_row[n] = acc;
    }
  }
}

void matmul_neon(const float *A, const float *B, float *C, int M, int N,
                 int K) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  for (int m = 0; m < M; ++m) {
    int n = 0;
    for (; n + 4 <= N; n += 4) {
      float32x4_t acc = vdupq_n_f32(0.0f);
      for (int k = 0; k < K; ++k) {
        const float32x4_t b_vec = vld1q_f32(B + k * N + n);
        const float32x4_t a_val = vdupq_n_f32(A[m * K + k]);
        acc = vmlaq_f32(acc, a_val, b_vec);
      }
      vst1q_f32(C + m * N + n, acc);
    }
    for (; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = acc;
    }
  }
#else
  matmul_naive(A, B, C, M, N, K);
#endif
}

void matmul_skinny(const float *A, const float *B, float *C, int M, int N,
                   int K) {
  const float *b0 = B;
  const float *b1 = B + N;
  for (int m = 0; m < M; ++m) {
    const float *a_row = A + m * K;
    float a0 = a_row[0];
    float a1 = a_row[1];
    float *c_row = C + m * N;
    UNROLL4
    for (int n = 0; n < N; ++n) {
      c_row[n] = a0 * b0[n] + a1 * b1[n];
    }
  }
}

void matmul_skinny_neon(const float *A, const float *B, float *C, int M, int N,
                        int K) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float *b0 = B;
  const float *b1 = B + N;
  for (int m = 0; m < M; ++m) {
    const float *a_row = A + m * K;
    const float a0 = a_row[0];
    const float a1 = a_row[1];
    float *c_row = C + m * N;
    int n = 0;
    const float32x4_t a0_vec = vdupq_n_f32(a0);
    const float32x4_t a1_vec = vdupq_n_f32(a1);
    for (; n + 4 <= N; n += 4) {
      const float32x4_t b0_vec = vld1q_f32(b0 + n);
      const float32x4_t b1_vec = vld1q_f32(b1 + n);
      float32x4_t acc = vmulq_f32(a0_vec, b0_vec);
      acc = vmlaq_f32(acc, a1_vec, b1_vec);
      vst1q_f32(c_row + n, acc);
    }
    for (; n < N; ++n) {
      c_row[n] = a0 * b0[n] + a1 * b1[n];
    }
  }
#else
  matmul_skinny(A, B, C, M, N, K);
#endif
}

template <int TILE_SIZE>
void matmul_tiled(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  for (int m0 = 0; m0 < M; m0 += TILE_SIZE) {
    int m_max = std::min(m0 + TILE_SIZE, M);
    for (int n0 = 0; n0 < N; n0 += TILE_SIZE) {
      int n_max = std::min(n0 + TILE_SIZE, N);
      int n_block = n_max - n0;
      for (int m = m0; m < m_max; ++m) {
        float accum[TILE_SIZE];
        zero_buffer(accum, static_cast<size_t>(n_block));
        for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
          int k_max = std::min(k0 + TILE_SIZE, K);
          for (int k = k0; k < k_max; ++k) {
            float a_val = A[m * K + k];
            const float *b_ptr = B + k * N + n0;
            for (int ni = 0; ni < n_block; ++ni) {
              accum[ni] += a_val * b_ptr[ni];
            }
          }
        }
        float *c_row = C + m * N + n0;
        for (int ni = 0; ni < n_block; ++ni) {
          c_row[ni] = accum[ni];
        }
      }
    }
  }
}

template <int TILE_SIZE>
void matmul_tiled_neon(const float *A, const float *B, float *C, int M, int N,
                       int K) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  matmul_tiled<TILE_SIZE>(A, B, C, M, N, K);
#else
  matmul_tiled<TILE_SIZE>(A, B, C, M, N, K);
#endif
}

// TODO - First benchmark and then use M-Series `Accelerate` Matmul kernel
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
  constexpr int TILE = 256;
  MatmulKernel kernel = predict_matmul_kernel(M, K, N);
  if (kernel == MatmulKernel::Skinny && K != 2) kernel = MatmulKernel::Naive;

  switch (kernel) {
    case MatmulKernel::Skinny:
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      matmul_skinny_neon(A, B, C, M, N, K);
#else
      matmul_skinny(A, B, C, M, N, K);
#endif
      break;
    case MatmulKernel::Naive:
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      matmul_neon(A, B, C, M, N, K);
#else
      matmul_naive(A, B, C, M, N, K);
#endif
      break;
    case MatmulKernel::Tiled:
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      matmul_tiled_neon<TILE>(A, B, C, M, N, K);
#else
      matmul_tiled<TILE>(A, B, C, M, N, K);
#endif
      break;
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

#undef UNROLL4
