#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "bench_runner.hpp"

namespace {

BenchmarkConfig parse_flags(int argc, char** argv, int& random_samples) {
  BenchmarkConfig cfg{512, 512, 512, 10};
  random_samples = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("M=", 0) == 0) {
      cfg.M = std::atoi(arg.c_str() + 2);
    } else if (arg.rfind("K=", 0) == 0) {
      cfg.K = std::atoi(arg.c_str() + 2);
    } else if (arg.rfind("N=", 0) == 0) {
      cfg.N = std::atoi(arg.c_str() + 2);
    } else if (arg.rfind("I=", 0) == 0) {
      cfg.iterations = std::atoi(arg.c_str() + 2);
    } else if (arg.rfind("R=", 0) == 0) {
      random_samples = std::atoi(arg.c_str() + 2);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(1);
    }
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    std::vector<std::pair<std::string, BenchmarkConfig>> defaults = {
        {"skinny-small", BenchmarkConfig{64, 2, 10, 1000}},
        {"skinny-large", BenchmarkConfig{20000, 2, 10, 100}},
        {"square-small", BenchmarkConfig{128, 128, 128, 10}},
        {"square-large", BenchmarkConfig{512, 512, 512, 5}},
        {"exp-1", BenchmarkConfig{64, 10, 64, 10}},
        {"exp-2", BenchmarkConfig{20000, 10, 5, 10}},
        {"exp-3", BenchmarkConfig{64, 5, 1, 10}},
    };
    std::cout
        << "No dimensions provided. Running default matmul benchmark suite..."
        << std::endl;
    for (const auto& entry : defaults) {
      run_benchmarks(entry.second, entry.first);
      std::cout << std::endl;
    }
    return 0;
  }

  int random_samples = 0;
  auto cfg = parse_flags(argc, argv, random_samples);
  if (cfg.M <= 0 || cfg.K <= 0 || cfg.N <= 0) {
    std::cerr << "Dimensions must be positive." << std::endl;
    return 1;
  }
  if (cfg.iterations <= 0) cfg.iterations = 1;

  if (random_samples > 0) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist_mn(1, 512);
    std::uniform_int_distribution<int> dist_k(1, 128);
    std::cout << "Random trials: " << random_samples << std::endl;
    for (int i = 0; i < random_samples; ++i) {
      BenchmarkConfig sample{dist_mn(rng), dist_k(rng), dist_mn(rng),
                             std::max(1, cfg.iterations)};
      BenchmarkSummary summary = collect_benchmark(sample);
      std::cout << "  [" << i + 1 << "] M=" << summary.config.M
                << " K=" << summary.config.K << " N=" << summary.config.N
                << " best=" << summary.actual << std::endl;
    }
    return 0;
  }

  run_benchmarks(cfg);
  return 0;
}
