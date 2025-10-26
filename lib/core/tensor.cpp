#include "tensor.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
size_t compute_numel(const std::vector<int>& shape) {
  size_t n = 1;
  for (int d : shape) {
    if (d <= 0) throw std::invalid_argument("Tensor shape must be positive");
    n *= static_cast<size_t>(d);
  }
  return n;
}

inline void zero_buffer(float* ptr, size_t count) {
  if (!ptr || count == 0) return;
  std::memset(ptr, 0, count * sizeof(float));
}
}  // namespace

namespace {

void backward_add(TapeOp& op) {
  const float* g_out = op.out.grad();
  float* ga = op.a.grad();
  float* gb = op.b.grad();
  if (!g_out || !ga || !gb) return;
  vDSP_Length len = static_cast<vDSP_Length>(op.out.numel);
  vDSP_vadd(ga, 1, g_out, 1, ga, 1, len);
  vDSP_vadd(gb, 1, g_out, 1, gb, 1, len);
}

void backward_sub(TapeOp& op) {
  const float* g_out = op.out.grad();
  float* ga = op.a.grad();
  float* gb = op.b.grad();
  if (!g_out || !ga || !gb) return;
  vDSP_Length len = static_cast<vDSP_Length>(op.out.numel);
  vDSP_vadd(ga, 1, g_out, 1, ga, 1, len);
  vDSP_vsub(g_out, 1, gb, 1, gb, 1, len);
}

void backward_mul(TapeOp& op) {
  const float* g_out = op.out.grad();
  const float* a_data = op.a.data();
  const float* b_data = op.b.data();
  float* ga = op.a.grad();
  float* gb = op.b.grad();
  if (!g_out || !a_data || !b_data || !ga || !gb) return;
  vDSP_Length len = static_cast<vDSP_Length>(op.out.numel);
  std::vector<float> scratch(op.out.numel);
  vDSP_vmul(g_out, 1, b_data, 1, scratch.data(), 1, len);
  vDSP_vadd(ga, 1, scratch.data(), 1, ga, 1, len);
  vDSP_vmul(g_out, 1, a_data, 1, scratch.data(), 1, len);
  vDSP_vadd(gb, 1, scratch.data(), 1, gb, 1, len);
}

void backward_relu(TapeOp& op) {
  const float* g_out = op.out.grad();
  const float* x = op.a.data();
  float* gx = op.a.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    gx[i] += g_out[i] * (x[i] > 0.0f ? 1.0f : 0.0f);
  }
}

void backward_tanh(TapeOp& op) {
  const float* g_out = op.out.grad();
  const float* y = op.out.data();
  float* gx = op.a.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    gx[i] += g_out[i] * (1.0f - y[i] * y[i]);
  }
}

void backward_sigmoid(TapeOp& op) {
  const float* g_out = op.out.grad();
  const float* y = op.out.data();
  float* gx = op.a.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    gx[i] += g_out[i] * y[i] * (1.0f - y[i]);
  }
}

void backward_log(TapeOp& op) {
  const float* g_out = op.out.grad();
  const float* x = op.a.data();
  float* gx = op.a.grad();
  for (size_t i = 0; i < op.out.numel; ++i) {
    gx[i] += g_out[i] / x[i];
  }
}

void backward_sum(TapeOp& op) {
  const float* g_out_ptr = op.out.grad();
  float* gx = op.a.grad();
  if (!g_out_ptr || !gx) return;
  const float g_out = g_out_ptr[0];
  vDSP_Length len = static_cast<vDSP_Length>(op.a.numel);
  vDSP_vsadd(gx, 1, &g_out, gx, 1, len);
}

// Use Accelerate-backed GEMM for matmul gradients.
void backward_matmul(TapeOp& op) {
  int M = op.a.shape[0];
  int K = op.a.shape[1];
  int N = op.b.shape[1];
  const float* A = op.a.data();
  const float* B = op.b.data();
  const float* gY = op.out.grad();
  float* gA = op.a.grad();
  float* gB = op.b.grad();

  const float alpha = 1.0f;
  const float beta = 1.0f;
  const __LAPACK_int m = static_cast<__LAPACK_int>(M);
  const __LAPACK_int n = static_cast<__LAPACK_int>(N);
  const __LAPACK_int k = static_cast<__LAPACK_int>(K);
  const __LAPACK_int ldgy = n;
  const __LAPACK_int ldb = n;
  const __LAPACK_int ldga = k;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, n, alpha, gY, ldgy,
              B, ldb, beta, gA, ldga);

  const __LAPACK_int lda = k;
  const __LAPACK_int ldgb = n;
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, n, m, alpha, A, lda,
              gY, ldgy, beta, gB, ldgb);
}

void backward_add_rowwise(TapeOp& op) {
  int N = op.a.shape[0];
  int H = op.a.shape[1];
  const float* g_out = op.out.grad();
  float* gX = op.a.grad();
  float* gb = op.b.grad();
  if (!g_out || !gX || !gb) return;
  vDSP_Length total = static_cast<vDSP_Length>(N * H);
  vDSP_vadd(gX, 1, g_out, 1, gX, 1, total);
  for (int h = 0; h < H; ++h) {
    float acc = 0.0f;
    vDSP_Length len = static_cast<vDSP_Length>(N);
    vDSP_sve(g_out + h, H, &acc, len);
    gb[h] += acc;
  }
}

}  // namespace

// Tensor methods
float* Tensor::data() { return store ? store->data_ptr(offset) : nullptr; }
float* Tensor::grad() { return store ? store->grad_ptr(offset) : nullptr; }
const float* Tensor::data() const {
  return store ? store->data_ptr(offset) : nullptr;
}
const float* Tensor::grad() const {
  return store ? store->grad_ptr(offset) : nullptr;
}

void Tensor::zero_grad() {
  if (!store) return;
  float* ptr = store->grad_ptr(offset);
  if (!ptr) return;
  zero_buffer(ptr, numel);
}

void Tensor::fill(float v) {
  if (!store) return;
  float* ptr = store->data_ptr(offset);
  if (!ptr) return;
  std::fill(ptr, ptr + numel, v);
}

void ParameterStore::register_parameter_allocation(size_t offset,
                                                   size_t count) {
  if (count == 0) return;
  if (!param_block_initialized) {
    param_grad_offset = offset;
    param_grad_span = count;
    param_grad_elements = count;
    param_block_initialized = true;
    param_block_contiguous = true;
    return;
  }

  if (offset < param_grad_offset) {
    param_block_contiguous = false;
    size_t new_end = param_grad_offset + param_grad_span;
    param_grad_offset = offset;
    param_grad_span = new_end - param_grad_offset;
  }

  const size_t block_end = param_grad_offset + param_grad_span;
  if (offset != block_end) {
    param_block_contiguous = false;
  }

  const size_t new_end = offset + count;
  if (new_end > param_grad_offset + param_grad_span) {
    param_grad_span = new_end - param_grad_offset;
  }
  param_grad_elements += count;
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

size_t ParameterStore::mark() const { return used; }

void ParameterStore::reset(size_t mark) {
  if (mark > used)
    throw std::invalid_argument("ParameterStore::reset mark beyond used");
  used = mark;
  if (stats_enabled) {
    stats.peak_elements = std::max(stats.peak_elements, used);
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

float* ParameterStore::data_ptr(size_t offset) {
  return data_buf ? data_buf.get() + offset : nullptr;
}

const float* ParameterStore::data_ptr(size_t offset) const {
  return data_buf ? data_buf.get() + offset : nullptr;
}

float* ParameterStore::grad_ptr(size_t offset) {
  return grad_buf ? grad_buf.get() + offset : nullptr;
}

const float* ParameterStore::grad_ptr(size_t offset) const {
  return grad_buf ? grad_buf.get() + offset : nullptr;
}

Tensor ParameterStore::tensor(const std::vector<int>& shape, TensorInit init) {
  const bool zero_data = (init == TensorInit::ZeroData);
  const size_t n = compute_numel(shape);
  const size_t off = allocate(n);

  if (n == 0) return Tensor{this, off, shape, n};

  float* data = data_ptr(off);
  float* grad = grad_ptr(off);

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

Tensor ParameterStore::parameter(const std::vector<int>& shape, float scale,
                                 unsigned seed) {
  auto t = tensor(shape);
  if (t.numel > 0) {
    register_parameter_allocation(t.offset, t.numel);
  }
  std::uniform_real_distribution<float> dist(-scale, scale);
  auto* p = t.data();
  if (seed == 0) {
    for (size_t i = 0; i < t.numel; ++i) {
      p[i] = dist(rng);
    }
  } else {
    std::mt19937 gen(seed);
    for (size_t i = 0; i < t.numel; ++i) {
      p[i] = dist(gen);
    }
  }
  return t;
}

void ParameterStore::seed(unsigned seed) { rng.seed(seed); }

void ParameterStore::enable_stats(bool enabled) {
  stats_enabled = enabled;
  reset_stats();
}

void ParameterStore::reset_stats() { stats = ParameterStoreStats{}; }

const ParameterStoreStats& ParameterStore::get_stats() const { return stats; }

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
  size_t zero_offset = 0;
  size_t zero_count = 0;

  if (param_block_initialized) {
    zero_offset = param_grad_offset;
    zero_count = param_grad_span;
  } else {
    zero_offset = 0;
    zero_count = used;
  }

#ifndef NDEBUG
  if (param_block_initialized && param_block_contiguous) {
    assert(param_grad_span == param_grad_elements);
  }
#endif

  if (zero_count == 0) {
    if (stats_enabled) {
      stats.zero_grad_calls += 1;
    }
    return;
  }

  float* grad_base = grad_ptr(zero_offset);
  if (!grad_base) {
    if (stats_enabled) {
      stats.zero_grad_calls += 1;
    }
    return;
  }

  if (stats_enabled) {
    auto start = std::chrono::steady_clock::now();
    zero_buffer(grad_base, zero_count);
    auto end = std::chrono::steady_clock::now();
    stats.zero_grad_calls += 1;
    stats.zero_grad_elems += zero_count;
    stats.zero_grad_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
  } else {
    zero_buffer(grad_base, zero_count);
  }
}

void ParameterStore::clear_tape() { tape.clear(); }

void ParameterStore::backward(const Tensor& loss) {
  if (loss.store != this)
    throw std::invalid_argument("loss belongs to different store");
  // Seed dL/dL = 1
  float* g = loss.store->grad_ptr(loss.offset);
  if (loss.numel == 1) {
    g[0] += 1.0f;
  } else {
    for (size_t i = 0; i < loss.numel; ++i) g[i] += 1.0f;
  }
  // Traverse tape in reverse
  for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
    TapeOp& op = *it;
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
static void assert_same_shape(const Tensor& a, const Tensor& b) {
  if (a.shape.size() != b.shape.size())
    throw std::invalid_argument("Shape rank mismatch");
  if (a.numel != b.numel) throw std::invalid_argument("Numel mismatch");
  for (size_t i = 0; i < a.shape.size(); ++i)
    if (a.shape[i] != b.shape[i]) throw std::invalid_argument("Shape mismatch");
}

Tensor add(const Tensor& a, const Tensor& b, ParameterStore& store) {
  assert_same_shape(a, b);
  Tensor out = store.tensor(a.shape);
  const float* ap = a.data();
  const float* bp = b.data();
  float* op = out.data();
  vDSP_Length len = static_cast<vDSP_Length>(a.numel);
  vDSP_vadd(ap, 1, bp, 1, op, 1, len);
  store.tape.push_back(TapeOp{OpType::Add, out, a, b});
  return out;
}

Tensor sub(const Tensor& a, const Tensor& b, ParameterStore& store) {
  assert_same_shape(a, b);
  Tensor out = store.tensor(a.shape);
  const float* ap = a.data();
  const float* bp = b.data();
  float* op = out.data();
  vDSP_Length len = static_cast<vDSP_Length>(a.numel);
  vDSP_vsub(bp, 1, ap, 1, op, 1, len);
  store.tape.push_back(TapeOp{OpType::Sub, out, a, b});
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b, ParameterStore& store) {
  assert_same_shape(a, b);
  Tensor out = store.tensor(a.shape);
  const float* ap = a.data();
  const float* bp = b.data();
  float* op = out.data();
  vDSP_Length len = static_cast<vDSP_Length>(a.numel);
  vDSP_vmul(ap, 1, bp, 1, op, 1, len);
  store.tape.push_back(TapeOp{OpType::Mul, out, a, b});
  return out;
}

Tensor relu(const Tensor& x, ParameterStore& store) {
  Tensor out = store.tensor(x.shape);
  const float* xp = x.data();
  float* op = out.data();
  if (!xp || !op) return out;
  vDSP_Length len = static_cast<vDSP_Length>(x.numel);
  float threshold = 0.0f;
  vDSP_vthres(xp, 1, &threshold, op, 1, len);
  store.tape.push_back(TapeOp{OpType::Relu, out, x, Tensor{}});
  return out;
}

Tensor vtanh(const Tensor& x, ParameterStore& store) {
  Tensor out = store.tensor(x.shape);
  const float* xp = x.data();
  float* op = out.data();
  if (!xp || !op) return out;
  int len = static_cast<int>(x.numel);
  if (len > 0) vvtanhf(op, xp, &len);
  store.tape.push_back(TapeOp{OpType::Tanh, out, x, Tensor{}});
  return out;
}

Tensor sigmoid(const Tensor& x, ParameterStore& store) {
  Tensor out = store.tensor(x.shape);
  const float* xp = x.data();
  float* op = out.data();
  if (!xp || !op) return out;
  int len_int = static_cast<int>(x.numel);
  if (len_int > 0) {
    vDSP_Length len = static_cast<vDSP_Length>(x.numel);
    const float one = 1.0f;
    const float neg_one = -1.0f;
    vDSP_vsmul(xp, 1, &neg_one, op, 1, len);
    vvexpf(op, op, &len_int);
    vDSP_vsadd(op, 1, &one, op, 1, len);
    vDSP_svdiv(&one, op, 1, op, 1, len);
  }
  store.tape.push_back(TapeOp{OpType::Sigmoid, out, x, Tensor{}});
  return out;
}

Tensor vlog(const Tensor& x, ParameterStore& store) {
  Tensor out = store.tensor(x.shape);
  const float* xp = x.data();
  float* op = out.data();
  if (!xp || !op) return out;
  int len = static_cast<int>(x.numel);
  if (len > 0) vvlogf(op, xp, &len);
  store.tape.push_back(TapeOp{OpType::Log, out, x, Tensor{}});
  return out;
}

Tensor sum(const Tensor& x, ParameterStore& store) {
  Tensor out = store.tensor({1});
  const float* xp = x.data();
  float* op = out.data();
  if (!xp || !op) return out;
  vDSP_Length len = static_cast<vDSP_Length>(x.numel);
  float acc = 0.0f;
  if (len > 0) vDSP_sve(xp, 1, &acc, len);
  op[0] = acc;
  store.tape.push_back(TapeOp{OpType::Sum, out, x, Tensor{}});
  return out;
}

// Matmul uses Accelerate-backed GEMM exclusively.
Tensor matmul(const Tensor& a, const Tensor& b, ParameterStore& store) {
  if (a.shape.size() != 2 || b.shape.size() != 2)
    throw std::invalid_argument("matmul expects 2D tensors");
  int M = a.shape[0];
  int K = a.shape[1];
  int K2 = b.shape[0];
  int N = b.shape[1];
  if (K != K2) throw std::invalid_argument("matmul inner dim mismatch");
  Tensor out = store.tensor({M, N});

  const float* A = a.data();
  const float* B = b.data();
  float* C = out.data();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const __LAPACK_int m = static_cast<__LAPACK_int>(M);
  const __LAPACK_int n = static_cast<__LAPACK_int>(N);
  const __LAPACK_int k = static_cast<__LAPACK_int>(K);
  const __LAPACK_int lda = k;
  const __LAPACK_int ldb = n;
  const __LAPACK_int ldc = n;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda,
              B, ldb, beta, C, ldc);
  store.tape.push_back(TapeOp{OpType::Matmul, out, a, b});
  return out;
}

Tensor add_rowwise(const Tensor& X, const Tensor& b, ParameterStore& store) {
  if (X.shape.size() != 2 || b.shape.size() != 1)
    throw std::invalid_argument("add_rowwise expects X[N,H], b[H]");
  int N = X.shape[0];
  int H = X.shape[1];
  if (b.shape[0] != H) throw std::invalid_argument("add_rowwise dim mismatch");
  Tensor out = store.tensor({N, H});
  const float* xp = X.data();
  const float* bp = b.data();
  float* op = out.data();
  if (!xp || !bp || !op) return out;
  vDSP_Length row_len = static_cast<vDSP_Length>(H);
  for (int n = 0; n < N; ++n) {
    const float* x_row = xp + n * H;
    float* out_row = op + n * H;
    vDSP_vadd(x_row, 1, bp, 1, out_row, 1, row_len);
  }
  store.tape.push_back(TapeOp{OpType::AddRowwise, out, X, b});
  return out;
}
