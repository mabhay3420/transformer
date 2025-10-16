#pragma once

#include <algorithm>

enum class MatmulKernel { Naive, Tiled, Skinny };

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
  if (K == 2) return MatmulKernel::Skinny;
  if (M <= 0 || N <= 0 || K <= 0) return MatmulKernel::Naive;
  if (K <= 1) return MatmulKernel::Naive;

  const int max_dim = std::max({M, N, K});
  const int min_dim = std::min({M, N, K});

  const bool very_small = (max_dim <= 16 && min_dim <= 8);
  const bool small_rect = (N <= 16 && K <= 16);
  const bool very_skinny = (N <= 8);
  const bool tall_skinny = (M >= 512 && N <= 32);
  const bool medium_skinny = (N <= 12 && K <= 32);

  if (very_small || small_rect || very_skinny || tall_skinny || medium_skinny) {
    return MatmulKernel::Naive;
  }

  return MatmulKernel::Tiled;
}

inline MatmulKernel predict_matmul_kernel(const int shape[3]) {
  return predict_matmul_kernel(shape[0], shape[1], shape[2]);
}
