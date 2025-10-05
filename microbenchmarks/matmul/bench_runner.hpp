#pragma once

#include <functional>
#include <string>
#include <vector>

using MatmulFn = std::function<void(const float*, const float*, float*, int, int, int)>;

#include <string>
#include <vector>

struct MatmulBenchmark {
  std::string name;
  MatmulFn fn;
};

const std::vector<MatmulBenchmark>& get_matmul_benchmarks();

struct BenchmarkConfig {
  int M;
  int K;
  int N;
  int iterations;
};

struct BenchmarkResult {
  std::string name;
  double milliseconds;
};

struct BenchmarkSummary {
  BenchmarkConfig config;
  std::vector<BenchmarkResult> results;
  std::string predicted;
  std::string actual;
};

BenchmarkSummary collect_benchmark(const BenchmarkConfig& cfg);
void run_benchmarks(const BenchmarkConfig& cfg, const std::string& label = "");
