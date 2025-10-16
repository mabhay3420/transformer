#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace {

using clock = std::chrono::steady_clock;

using BinaryOp = void (*)(const float*, const float*, float*, size_t);
using UnaryOp = void (*)(const float*, float*, size_t);
using SumOp = float (*)(const float*, size_t);
using RowwiseOp = void (*)(const float*, const float*, float*, int, int);

volatile float sink = 0.0f;

void add_scalar(const float* a, const float* b, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void sub_scalar(const float* a, const float* b, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

void mul_scalar(const float* a, const float* b, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

void relu_scalar(const float* x, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

float sum_scalar(const float* x, size_t n) {
  float acc = 0.0f;
  for (size_t i = 0; i < n; ++i) acc += x[i];
  return acc;
}

void add_rowwise_scalar(const float* X, const float* b, float* out, int rows,
                        int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* x_row = X + r * cols;
    float* o_row = out + r * cols;
    for (int c = 0; c < cols; ++c) o_row[c] = x_row[c] + b[c];
  }
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

void add_neon(const float* a, const float* b, float* out, size_t n) {
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t vc = vaddq_f32(va, vb);
    vst1q_f32(out + i, vc);
  }
  for (; i < n; ++i) out[i] = a[i] + b[i];
}

void sub_neon(const float* a, const float* b, float* out, size_t n) {
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t vc = vsubq_f32(va, vb);
    vst1q_f32(out + i, vc);
  }
  for (; i < n; ++i) out[i] = a[i] - b[i];
}

void mul_neon(const float* a, const float* b, float* out, size_t n) {
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t vc = vmulq_f32(va, vb);
    vst1q_f32(out + i, vc);
  }
  for (; i < n; ++i) out[i] = a[i] * b[i];
}

void relu_neon(const float* x, float* out, size_t n) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t vx = vld1q_f32(x + i);
    float32x4_t vy = vmaxq_f32(vx, zero);
    vst1q_f32(out + i, vy);
  }
  for (; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

float sum_neon(const float* x, size_t n) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    acc = vaddq_f32(acc, vld1q_f32(x + i));
  }
  float sum = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
              vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
  for (; i < n; ++i) sum += x[i];
  return sum;
}

void add_rowwise_neon(const float* X, const float* b, float* out, int rows,
                      int cols) {
  int vecs = cols / 4;
  int rem = cols - vecs * 4;
  for (int r = 0; r < rows; ++r) {
    const float* x_row = X + r * cols;
    float* o_row = out + r * cols;
    int c = 0;
    for (; c < vecs; ++c) {
      float32x4_t xv = vld1q_f32(x_row + c * 4);
      float32x4_t bv = vld1q_f32(b + c * 4);
      vst1q_f32(o_row + c * 4, vaddq_f32(xv, bv));
    }
    for (int i = 0; i < rem; ++i) {
      o_row[vecs * 4 + i] = x_row[vecs * 4 + i] + b[vecs * 4 + i];
    }
  }
}

#endif

double time_binary(BinaryOp op, const std::vector<float>& a,
                   const std::vector<float>& b, std::vector<float>& out,
                   int iterations) {
  op(a.data(), b.data(), out.data(), a.size());
  auto start = clock::now();
  for (int i = 0; i < iterations; ++i)
    op(a.data(), b.data(), out.data(), a.size());
  auto end = clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  sink += out[0];
  return elapsed.count() / static_cast<double>(iterations);
}

double time_unary(UnaryOp op, const std::vector<float>& x,
                  std::vector<float>& out, int iterations) {
  op(x.data(), out.data(), x.size());
  auto start = clock::now();
  for (int i = 0; i < iterations; ++i) op(x.data(), out.data(), x.size());
  auto end = clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  sink += out[0];
  return elapsed.count() / static_cast<double>(iterations);
}

double time_sum(SumOp op, const std::vector<float>& x, int iterations) {
  float val = op(x.data(), x.size());
  auto start = clock::now();
  for (int i = 0; i < iterations; ++i) val = op(x.data(), x.size());
  auto end = clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  sink += val;
  return elapsed.count() / static_cast<double>(iterations);
}

double time_rowwise(RowwiseOp op, const std::vector<float>& X,
                    const std::vector<float>& b, std::vector<float>& out,
                    int rows, int cols, int iterations) {
  op(X.data(), b.data(), out.data(), rows, cols);
  auto start = clock::now();
  for (int i = 0; i < iterations; ++i)
    op(X.data(), b.data(), out.data(), rows, cols);
  auto end = clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  sink += out[0];
  return elapsed.count() / static_cast<double>(iterations);
}

void print_result(const std::string& label, double scalar_ms, double neon_ms) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  scalar: " << scalar_ms << " ms" << std::endl;
  if (neon_ms > 0.0) {
    double speedup = scalar_ms / neon_ms;
    std::cout << "  neon  : " << neon_ms << " ms (×" << speedup << ")"
              << std::endl;
  }
  std::cout.unsetf(std::ios::floatfield);
}

void run_binary_suite(const std::string& name, BinaryOp scalar_op,
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                      BinaryOp neon_op,
#else
                      BinaryOp,
#endif
                      size_t numel, int iterations) {
  std::vector<float> a(numel);
  std::vector<float> b(numel);
  std::vector<float> out(numel);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < numel; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
  }
  std::cout << "== " << name << " (N=" << numel << ", iters=" << iterations
            << ") ==" << std::endl;
  double scalar_ms = time_binary(scalar_op, a, b, out, iterations);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  double neon_ms = time_binary(neon_op, a, b, out, iterations);
#else
  double neon_ms = 0.0;
#endif
  print_result(name, scalar_ms, neon_ms);
  std::cout << std::endl;
}

void run_unary_suite(const std::string& name, UnaryOp scalar_op,
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                     UnaryOp neon_op,
#else
                     UnaryOp,
#endif
                     size_t numel, int iterations) {
  std::vector<float> x(numel);
  std::vector<float> out(numel);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < numel; ++i) x[i] = dist(rng);
  std::cout << "== " << name << " (N=" << numel << ", iters=" << iterations
            << ") ==" << std::endl;
  double scalar_ms = time_unary(scalar_op, x, out, iterations);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  double neon_ms = time_unary(neon_op, x, out, iterations);
#else
  double neon_ms = 0.0;
#endif
  print_result(name, scalar_ms, neon_ms);
  std::cout << std::endl;
}

void run_sum_suite(size_t numel, int iterations) {
  std::vector<float> x(numel);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < numel; ++i) x[i] = dist(rng);
  std::cout << "== sum (N=" << numel << ", iters=" << iterations
            << ") ==" << std::endl;
  double scalar_ms = time_sum(sum_scalar, x, iterations);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  double neon_ms = time_sum(sum_neon, x, iterations);
#else
  double neon_ms = 0.0;
#endif
  print_result("sum", scalar_ms, neon_ms);
  std::cout << std::endl;
}

void run_rowwise_suite(int rows, int cols, int iterations) {
  const size_t total = static_cast<size_t>(rows) * cols;
  std::vector<float> X(total);
  std::vector<float> b(cols);
  std::vector<float> out(total);
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < total; ++i) X[i] = dist(rng);
  for (int i = 0; i < cols; ++i) b[i] = dist(rng);
  std::cout << "== add_rowwise (rows=" << rows << ", cols=" << cols
            << ", iters=" << iterations << ") ==" << std::endl;
  double scalar_ms =
      time_rowwise(add_rowwise_scalar, X, b, out, rows, cols, iterations);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  double neon_ms =
      time_rowwise(add_rowwise_neon, X, b, out, rows, cols, iterations);
#else
  double neon_ms = 0.0;
#endif
  print_result("add_rowwise", scalar_ms, neon_ms);
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  size_t numel = 1 << 20;
  int iterations = 200;
  if (argc > 1) numel = static_cast<size_t>(std::stoul(argv[1]));
  if (argc > 2) iterations = std::stoi(argv[2]);

  run_binary_suite("add", add_scalar,
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                   add_neon,
#else
                   nullptr,
#endif
                   numel, iterations);

  run_binary_suite("sub", sub_scalar,
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                   sub_neon,
#else
                   nullptr,
#endif
                   numel, iterations);

  run_binary_suite("mul", mul_scalar,
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                   mul_neon,
#else
                   nullptr,
#endif
                   numel, iterations);

  run_unary_suite("relu", relu_scalar,
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                  relu_neon,
#else
                  nullptr,
#endif
                  numel, iterations);

  run_sum_suite(numel, iterations);

  run_rowwise_suite(1024, 256, 200);

  return static_cast<int>(sink);
}
