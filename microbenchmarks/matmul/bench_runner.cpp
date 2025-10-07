#include "bench_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include "matmul_cost_model.hpp"

namespace {

std::vector<float> make_random_matrix(int rows, int cols) {
  std::vector<float> mat(static_cast<size_t>(rows) * cols);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : mat) v = dist(rng);
  return mat;
}

double run_single(const MatmulBenchmark& bench, const BenchmarkConfig& cfg) {
  const int M = cfg.M;
  const int K = cfg.K;
  const int N = cfg.N;
  auto A = make_random_matrix(M, K);
  auto B = make_random_matrix(K, N);
  std::vector<float> C(static_cast<size_t>(M) * N);

  // warmup
  bench.fn(A.data(), B.data(), C.data(), M, K, N);

  using clock = std::chrono::steady_clock;
  auto start = clock::now();
  for (int it = 0; it < cfg.iterations; ++it) {
    bench.fn(A.data(), B.data(), C.data(), M, K, N);
  }
  auto end = clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(cfg.iterations);
}

}  // namespace

BenchmarkSummary collect_benchmark(const BenchmarkConfig& cfg) {
  const auto& benches = get_matmul_benchmarks();
  BenchmarkSummary summary;
  summary.config = cfg;
  summary.results.reserve(benches.size());

  for (const auto& bench : benches) {
    if ((bench.name == "skinny_specialized" ||
         bench.name == "skinny_specialized_neon") &&
        cfg.K != 2)
      continue;
    double ms = run_single(bench, cfg);
    summary.results.push_back({bench.name, ms});
  }

  double baseline = 0.0;
  for (const auto& result : summary.results) {
    if (result.name == "naive") {
      baseline = result.milliseconds;
      break;
    }
  }
  if (baseline <= 0.0 && !summary.results.empty()) {
    baseline = summary.results.front().milliseconds;
  }

  summary.predicted = matmul_kernel_name(
      predict_matmul_kernel(cfg.M, cfg.K, cfg.N));
  summary.actual = summary.results.empty() ? std::string{}
                                           : summary.results.front().name;
  double best_ms = summary.results.empty() ? 0.0 :
                                              summary.results.front().milliseconds;
  for (const auto& result : summary.results) {
    if (result.milliseconds < best_ms) {
      best_ms = result.milliseconds;
      summary.actual = result.name;
    }
  }

  return summary;
}

void run_benchmarks(const BenchmarkConfig& cfg, const std::string& label) {
  if (!label.empty()) {
    std::cout << "[" << label << "]" << std::endl;
  }
  BenchmarkSummary summary = collect_benchmark(cfg);
  std::cout << "Benchmarking matmul implementations: " << std::endl;
  std::cout << "Dimensions: M=" << cfg.M << " K=" << cfg.K << " N=" << cfg.N
            << ", iterations=" << cfg.iterations << std::endl;

  double baseline = 0.0;
  for (const auto& result : summary.results) {
    if (result.name == "naive") {
      baseline = result.milliseconds;
      break;
    }
  }
  if (baseline <= 0.0 && !summary.results.empty()) {
    baseline = summary.results.front().milliseconds;
  }

  for (const auto& result : summary.results) {
    double factor = baseline > 0.0 && result.milliseconds > 0.0
                        ? baseline / result.milliseconds
                        : 0.0;
    std::cout << "  " << result.name << ": " << result.milliseconds << " ms";
    if (baseline > 0.0) {
      std::cout << " (×" << factor << ")";
    }
    if (result.name == summary.predicted) std::cout << " [predicted]";
    if (result.name == summary.actual) std::cout << " [best]";
    std::cout << std::endl;
  }

  auto get_ms = [&](const std::string& name) {
    for (const auto& result : summary.results) {
      if (result.name == name) return result.milliseconds;
    }
    return 0.0;
  };

  struct Pair {
    const char* scalar;
    const char* neon;
  };

  constexpr Pair neon_pairs[] = {
      {"naive", "naive_neon"},
      {"tiled", "tiled_neon"},
      {"skinny_specialized", "skinny_specialized_neon"},
  };

  bool printed_header = false;
  for (const auto& pair : neon_pairs) {
    const double scalar_ms = get_ms(pair.scalar);
    const double neon_ms = get_ms(pair.neon);
    if (scalar_ms <= 0.0 || neon_ms <= 0.0) continue;
    if (!printed_header) {
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "Neon speedups:" << std::endl;
      printed_header = true;
    }
    double speedup = scalar_ms / neon_ms;
    std::cout << "  " << pair.scalar << " → " << pair.neon << ": ×" << speedup
              << std::endl;
  }
  if (printed_header) {
    std::cout.unsetf(std::ios::floatfield);
    std::cout.precision(6);
  }

  std::cout << "Predicted best: " << summary.predicted << std::endl;
  std::cout << "Actual best   : " << summary.actual << std::endl;
}
