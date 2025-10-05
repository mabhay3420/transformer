#pragma once

#include <algorithm>
#include <limits>

enum class MatmulKernel {
  Naive,
  Tiled,
  Skinny
};

inline const char* matmul_kernel_name(MatmulKernel k) {
  switch (k) {
    case MatmulKernel::Naive:
      return "naive";
    case MatmulKernel::Tiled:
      return "tiled";
    case MatmulKernel::Skinny:
      return "skinny_specialized";
  }
  return "naive";
}

inline MatmulKernel predict_matmul_kernel(int M, int K, int N) {
  if (M <= 0 || K <= 0 || N <= 0) return MatmulKernel::Naive;

  const double flop = static_cast<double>(M) * K * N;
  const double bytes = 4.0 *
                       (static_cast<double>(M) * K + static_cast<double>(K) * N +
                        static_cast<double>(M) * N);

  constexpr double compute_naive = 16.0;   // relative flop throughput
  constexpr double compute_tiled = 64.0;
  constexpr double compute_skinny = 48.0;
  constexpr double bandwidth = 32.0;       // relative memory bandwidth

  double cost_naive = flop / compute_naive + bytes / bandwidth;

  // Naive suffers when K grows (poor reuse of B) but benefits when K is tiny.
  if (K > 32) cost_naive *= 1.5;
  if (K > 64) cost_naive *= 2.0;
  if (K <= 4) cost_naive *= 0.85;

  double cost_tiled = flop / compute_tiled + bytes / (bandwidth * 1.2);
  const int max_dim = std::max({M, N, K});
  const int min_dim = std::min({M, N, K});

  // Blocking overhead for very small problems: less compute to amortize packing.
  if (max_dim <= 64) cost_tiled *= 1.2;
  if (max_dim <= 32) cost_tiled *= 1.5;

  // Extremely skinny or short matrices increase kernel launch/setup costs.
  if (N <= 16 || M <= 16) cost_tiled *= 1.3;
  if (N <= 8) cost_tiled *= 2.5;
  if (N <= 8 && K >= 32) cost_tiled *= 1.5;
  if (M >= 512 && N <= 16) cost_tiled *= 1.6;

  double cost_skinny = std::numeric_limits<double>::infinity();
  if (K == 2) {
    // Two-column matmul streams each column of B once and reuses row factors from A.
    cost_skinny = flop / compute_skinny + bytes / (bandwidth * 1.5);
    if (M >= 2048) cost_skinny *= 0.9;  // large batch amortises loop control
  }

  if (cost_tiled < cost_naive) {
    if (K == 2 && cost_skinny < cost_tiled) return MatmulKernel::Skinny;
    return MatmulKernel::Tiled;
  }

  if (K == 2 && cost_skinny < cost_naive) return MatmulKernel::Skinny;
  return MatmulKernel::Naive;
}

inline MatmulKernel predict_matmul_kernel(const int shape[3]) {
  return predict_matmul_kernel(shape[0], shape[1], shape[2]);
}
