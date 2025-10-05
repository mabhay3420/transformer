#include "bench_runner.hpp"

#include <algorithm>

namespace {

void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
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

constexpr int TILE = 32;

void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
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
            for (int ni = 0; ni < n_block; ++ni) {
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

void matmul_skinny(const float* A, const float* B, float* C, int M, int K, int N) {
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

const std::vector<MatmulBenchmark>& registry() {
  static std::vector<MatmulBenchmark> benches = {
      {"naive", matmul_naive},
      {"tiled", matmul_tiled},
      {"skinny_specialized", matmul_skinny},
  };
  return benches;
}

}  // namespace

const std::vector<MatmulBenchmark>& get_matmul_benchmarks() { return registry(); }
