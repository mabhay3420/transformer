#include <algorithm>
#include "bench_runner.hpp"
#include "mlx/device.h"
#include "mlx/mlx.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace {

namespace mx = mlx::core;
void matmul_naive(const float* A, const float* B, float* C, int M, int K,
                  int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = acc;
    }
  }
}

template <int TILE>
void matmul_tiled_v3(const float* A, const float* B, float* C, int M, int K,
                     int N) {
  for (int m = 0; m < M; m += TILE) {
    int m_max = std::min(m + TILE, M);
    for (int n = 0; n < N; n += TILE) {
      int n_max = std::min(n + TILE, N);
      for (int mi = m; mi < m_max; ++mi) {
        float accum[TILE] = {0.0f};
        for (int k = 0; k < K; k += 1) {
          for (int ni = n; ni < n_max; ++ni) {
            accum[ni - n] += A[mi * K + k] * B[k * N + ni];
          }
        }
        for (int ni = n; ni < n_max; ++ni) {
          C[mi * N + ni] = accum[ni - n];
        }
      }
    }
  }
}

void matmul_skinny(const float* A, const float* B, float* C, int M, int K,
                   int N) {
  if (K != 2) {
    matmul_naive(A, B, C, M, K, N);
    return;
  }

  const int strideA = K;
  const int strideC = N;
  const float* b0 = B;
  const float* b1 = B + N;

  for (int m = 0; m < M; ++m) {
    const float* a_row = A + m * strideA;
    const float a0 = a_row[0];
    const float a1 = a_row[1];
    float* c_row = C + m * strideC;
    for (int n = 0; n < N; ++n) {
      c_row[n] = a0 * b0[n] + a1 * b1[n];
    }
  }
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

void matmul_naive_neon(const float* A, const float* B, float* C, int M, int K,
                       int N) {
  for (int m = 0; m < M; ++m) {
    int n = 0;
    for (; n + 4 <= N; n += 4) {
      float32x4_t acc = vdupq_n_f32(0.0f);
      for (int k = 0; k < K; ++k) {
        const float32x4_t b_vec = vld1q_f32(B + k * N + n);
        const float32x4_t a_vec = vdupq_n_f32(A[m * K + k]);
        acc = vmlaq_f32(acc, a_vec, b_vec);
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
}

template <int TILE>
void matmul_tiled_neon(const float* A, const float* B, float* C, int M, int K,
                       int N) {
  for (int m0 = 0; m0 < M; m0 += TILE) {
    int m_max = std::min(m0 + TILE, M);
    for (int n0 = 0; n0 < N; n0 += TILE) {
      int n_max = std::min(n0 + TILE, N);
      int n_block = n_max - n0;
      for (int m = m0; m < m_max; ++m) {
        float accum[TILE];
        for (int ni = 0; ni < n_block; ++ni) accum[ni] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += TILE) {
          int k_max = std::min(k0 + TILE, K);
          for (int k = k0; k < k_max; ++k) {
            const float a_val = A[m * K + k];
            const float* b_ptr = B + k * N + n0;
            int ni = 0;
            float32x4_t a_vec = vdupq_n_f32(a_val);
            for (; ni + 4 <= n_block; ni += 4) {
              float32x4_t acc_vec = vld1q_f32(accum + ni);
              float32x4_t b_vec = vld1q_f32(b_ptr + ni);
              acc_vec = vmlaq_f32(acc_vec, a_vec, b_vec);
              vst1q_f32(accum + ni, acc_vec);
            }
            for (; ni < n_block; ++ni) {
              accum[ni] += a_val * b_ptr[ni];
            }
          }
        }
        float* c_row = C + m * N + n0;
        for (int ni = 0; ni < n_block; ++ni) c_row[ni] = accum[ni];
      }
    }
  }
}

void matmul_skinny_specialized_neon(const float* A, const float* B, float* C,
                                    int M, int K, int N) {
  if (K != 2) {
    matmul_naive_neon(A, B, C, M, K, N);
    return;
  }

  const int strideA = K;
  const int strideC = N;
  const float* b0 = B;
  const float* b1 = B + N;

  for (int m = 0; m < M; ++m) {
    const float* a_row = A + m * strideA;
    const float a0 = a_row[0];
    const float a1 = a_row[1];
    float* c_row = C + m * strideC;
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
}

#endif

void matmul_mlx(const float* A, const float* B, float* C, int M, int K, int N) {
  mx::set_default_device(mx::Device::cpu);
  mx::array lhs(A, mx::Shape{M, K}, mx::float32);
  mx::array rhs(B, mx::Shape{K, N}, mx::float32);

  auto result = mx::matmul(lhs, rhs);
  result.eval();

  const float* result_ptr = result.data<float>();
  std::copy(result_ptr, result_ptr + static_cast<size_t>(M) * N, C);
}

const std::vector<MatmulBenchmark>& registry() {
  static std::vector<MatmulBenchmark> benches = {
    {"naive", matmul_naive},
    {"tiled_v3_256", matmul_tiled_v3<256>},
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    {"tiled_256_neon", matmul_tiled_neon<256>},
#endif
    {"mlx_auto", matmul_mlx},
  };
  return benches;
}

}  // namespace

const std::vector<MatmulBenchmark>& get_matmul_benchmarks() {
  return registry();
}
