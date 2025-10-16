#include "bench_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

#include "matmul_cost_model.hpp"

namespace {

struct AccuracyStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
};

std::vector<float> make_random_matrix(int rows, int cols) {
  std::vector<float> mat(static_cast<size_t>(rows) * cols);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : mat) v = dist(rng);
  return mat;
}

double run_single(const MatmulBenchmark& bench, const BenchmarkConfig& cfg,
                  const std::vector<float>& A, const std::vector<float>& B,
                  std::vector<float>& C) {
  const int M = cfg.M;
  const int K = cfg.K;
  const int N = cfg.N;

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

AccuracyStats compute_accuracy(const std::vector<float>& reference,
                               const std::vector<float>& candidate) {
  AccuracyStats stats{};
  if (reference.size() != candidate.size()) return stats;

  for (size_t i = 0; i < reference.size(); ++i) {
    const double ref = static_cast<double>(reference[i]);
    const double cand = static_cast<double>(candidate[i]);
    const double diff = std::abs(ref - cand);
    if (diff > stats.max_abs) stats.max_abs = diff;
    const double denom = std::abs(ref);
    if (denom > 1e-6) {
      const double rel = diff / denom;
      if (rel > stats.max_rel) stats.max_rel = rel;
    }
  }
  return stats;
}

std::string format_scientific(double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(3) << value;
  return oss.str();
}

}  // namespace

BenchmarkSummary collect_benchmark(const BenchmarkConfig& cfg) {
  const auto& benches = get_matmul_benchmarks();
  BenchmarkSummary summary;
  summary.config = cfg;
  summary.results.reserve(benches.size());

  const auto A = make_random_matrix(cfg.M, cfg.K);
  const auto B = make_random_matrix(cfg.K, cfg.N);

  struct RawResult {
    std::string name;
    double milliseconds;
    std::vector<float> output;
  };

  std::vector<RawResult> raw_results;
  raw_results.reserve(benches.size());

  for (const auto& bench : benches) {
    if ((bench.name == "skinny_specialized" ||
         bench.name == "skinny_specialized_neon") &&
        cfg.K != 2)
      continue;
    std::vector<float> C(static_cast<size_t>(cfg.M) * cfg.N);
    double ms = run_single(bench, cfg, A, B, C);
    raw_results.push_back(RawResult{bench.name, ms, std::move(C)});
  }

  summary.predicted =
      matmul_kernel_name(predict_matmul_kernel(cfg.M, cfg.K, cfg.N));

  if (raw_results.empty()) {
    summary.actual.clear();
    summary.reference.clear();
    return summary;
  }

  const RawResult* baseline = nullptr;
  for (const auto& result : raw_results) {
    if (result.name == "naive") {
      baseline = &result;
      break;
    }
  }
  if (!baseline) baseline = &raw_results.front();
  summary.reference = baseline->name;

  for (const auto& result : raw_results) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    if (&result != baseline) {
      const AccuracyStats stats =
          compute_accuracy(baseline->output, result.output);
      max_abs = stats.max_abs;
      max_rel = stats.max_rel;
    }
    summary.results.push_back(
        BenchmarkResult{result.name, result.milliseconds, max_abs, max_rel});
  }

  summary.actual = summary.results.front().name;
  double best_ms = summary.results.front().milliseconds;
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

  auto find_result = [&](const std::string& name) -> const BenchmarkResult* {
    for (const auto& result : summary.results) {
      if (result.name == name) return &result;
    }
    return nullptr;
  };

  const std::string reference_name =
      !summary.reference.empty()
          ? summary.reference
          : (!summary.results.empty() ? summary.results.front().name : "");
  const BenchmarkResult* baseline_result = find_result(reference_name);
  const double baseline = baseline_result && baseline_result->milliseconds > 0.0
                              ? baseline_result->milliseconds
                              : 0.0;

  for (const auto& result : summary.results) {
    double factor = baseline > 0.0 && result.milliseconds > 0.0
                        ? baseline / result.milliseconds
                        : 0.0;
    std::cout << "  " << result.name << ": " << result.milliseconds << " ms";
    if (baseline > 0.0 && result.milliseconds > 0.0) {
      std::cout << " (×" << factor << ")";
    }
    if (result.name == reference_name) std::cout << " [reference]";
    if (result.name == summary.predicted) std::cout << " [predicted]";
    if (result.name == summary.actual) std::cout << " [best]";
    if (result.name != reference_name) {
      if (result.max_abs_error > 0.0) {
        std::cout << " max|Δ|=" << format_scientific(result.max_abs_error);
        if (result.max_rel_error > 0.0) {
          std::cout << " max rel=" << format_scientific(result.max_rel_error);
        }
      } else {
        std::cout << " max|Δ|=0";
      }
    }
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
