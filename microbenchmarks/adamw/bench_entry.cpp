#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
  using value_type = T;
  using is_always_equal = std::true_type;
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() noexcept = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    void* ptr = ::operator new(n * sizeof(T), std::align_val_t(Alignment));
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t(Alignment));
  }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) {
  return false;
}

template <typename T>
T* assume_aligned(T* ptr) {
#if defined(__clang__) || defined(__GNUC__)
  return reinterpret_cast<T*>(__builtin_assume_aligned(ptr, 64));
#else
  return ptr;
#endif
}

template <typename T>
const T* assume_aligned(const T* ptr) {
#if defined(__clang__) || defined(__GNUC__)
  return reinterpret_cast<const T*>(__builtin_assume_aligned(ptr, 64));
#else
  return ptr;
#endif
}

struct AdamWConfig {
  int params = 32;
  int elements = 1 << 17;
  int iterations = 50;
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float weight_decay = 0.01f;
  bool use_weight_decay = true;
  bool amsgrad = false;
  float epsilon = 1e-8f;
  uint32_t seed = 1234;
};

struct Param {
  std::vector<float> data;
  std::vector<float> grad;
};

struct AdamWState {
  std::vector<std::vector<float>> m1;
  std::vector<std::vector<float>> m2;
  std::vector<std::vector<float>> vhat;
};

struct AdamWPacked {
  size_t params = 0;
  size_t elements = 0;
  float beta1_pow = 1.0f;
  float beta2_pow = 1.0f;
  using AlignedVector = std::vector<float, AlignedAllocator<float, 64>>;
  AlignedVector data;
  AlignedVector grad;
  AlignedVector m1;
  AlignedVector m2;
  AlignedVector vhat;
};

void ensure_state_size(std::vector<float>& state, size_t target) {
  if (state.size() != target) {
    state.assign(target, 0.0f);
  }
}

std::vector<Param> make_params(const AdamWConfig& cfg) {
  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<Param> params;
  params.reserve(static_cast<size_t>(cfg.params));
  for (int i = 0; i < cfg.params; ++i) {
    Param p;
    p.data.resize(static_cast<size_t>(cfg.elements));
    p.grad.resize(static_cast<size_t>(cfg.elements));
    for (int j = 0; j < cfg.elements; ++j) {
      p.data[static_cast<size_t>(j)] = dist(rng);
      p.grad[static_cast<size_t>(j)] = dist(rng);
    }
    params.push_back(std::move(p));
  }
  return params;
}

AdamWState make_state(const AdamWConfig& cfg) {
  AdamWState state;
  state.m1.resize(static_cast<size_t>(cfg.params));
  state.m2.resize(static_cast<size_t>(cfg.params));
  if (cfg.amsgrad) {
    state.vhat.resize(static_cast<size_t>(cfg.params));
  }
  return state;
}

AdamWPacked make_packed_params(const AdamWConfig& cfg) {
  AdamWPacked packed;
  packed.params = static_cast<size_t>(cfg.params);
  packed.elements = static_cast<size_t>(cfg.elements);
  const size_t total = packed.params * packed.elements;
  packed.data.resize(total);
  packed.grad.resize(total);
  packed.m1.assign(total, 0.0f);
  packed.m2.assign(total, 0.0f);
  if (cfg.amsgrad) {
    packed.vhat.assign(total, 0.0f);
  }

  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (size_t i = 0; i < total; ++i) {
    packed.data[i] = dist(rng);
    packed.grad[i] = dist(rng);
  }
  return packed;
}

using AdamWStep = void (*)(std::vector<Param>&, AdamWState&, size_t&,
                           const AdamWConfig&);
using AdamWPackedStep = void (*)(AdamWPacked&, size_t&, const AdamWConfig&);

template <bool UseDecay, bool UseAmsgrad>
inline void adamw_packed_update(float* data, const float* grad, float* m1,
                                float* m2, float* vhat, size_t total,
                                float beta1, float beta2, float one_minus_beta1,
                                float one_minus_beta2, float inv_bc1,
                                float inv_bc2, float weight_decay, float lr,
                                float epsilon) {
  size_t i = 0;
#define ADAMW_UPDATE_ONE(idx)                               \
  do {                                                      \
    float g = grad[idx];                                    \
    float m1_i = beta1 * m1[idx] + one_minus_beta1 * g;     \
    float m2_i = beta2 * m2[idx] + one_minus_beta2 * g * g; \
    m1[idx] = m1_i;                                         \
    m2[idx] = m2_i;                                         \
    float m1_hat = m1_i * inv_bc1;                          \
    float m2_hat;                                           \
    if constexpr (UseAmsgrad) {                             \
      float vhat_i = vhat[idx];                             \
      if (m2_i > vhat_i) vhat_i = m2_i;                     \
      vhat[idx] = vhat_i;                                   \
      m2_hat = vhat_i * inv_bc2;                            \
    } else {                                                \
      m2_hat = m2_i * inv_bc2;                              \
    }                                                       \
    float update = m1_hat / (std::sqrt(m2_hat) + epsilon);  \
    if constexpr (UseDecay) {                               \
      update += weight_decay * data[idx];                   \
    }                                                       \
    data[idx] -= lr * update;                               \
  } while (0)

  for (; i + 3 < total; i += 4) {
    ADAMW_UPDATE_ONE(i);
    ADAMW_UPDATE_ONE(i + 1);
    ADAMW_UPDATE_ONE(i + 2);
    ADAMW_UPDATE_ONE(i + 3);
  }
  for (; i < total; ++i) {
    ADAMW_UPDATE_ONE(i);
  }
#undef ADAMW_UPDATE_ONE
}

// Copied from include/optimizer.hpp to benchmark alternate implementations.
void adamw_step_baseline(std::vector<Param>& params, AdamWState& state,
                         size_t& step_count, const AdamWConfig& cfg) {
  const float lr = cfg.lr;
  ++step_count;
  const float bc1 = 1.0f - std::pow(cfg.beta1, static_cast<float>(step_count));
  const float bc2 = 1.0f - std::pow(cfg.beta2, static_cast<float>(step_count));

  for (size_t idx = 0; idx < params.size(); ++idx) {
    Param& param = params[idx];
    float* data = param.data.data();
    const float* grad_ptr = param.grad.data();
    const size_t n = param.data.size();
    ensure_state_size(state.m1[idx], n);
    ensure_state_size(state.m2[idx], n);
    if (cfg.amsgrad) {
      ensure_state_size(state.vhat[idx], n);
    }
    auto& m1_vec = state.m1[idx];
    auto& m2_vec = state.m2[idx];
    std::vector<float>* vhat_vec = cfg.amsgrad ? &state.vhat[idx] : nullptr;

    if (cfg.use_weight_decay && cfg.weight_decay != 0.0f) {
      for (size_t i = 0; i < n; ++i) {
        data[i] -= lr * cfg.weight_decay * data[i];
      }
    }

    if (cfg.amsgrad) {
      for (size_t i = 0; i < n; ++i) {
        float grad = grad_ptr[i];
        m1_vec[i] = cfg.beta1 * m1_vec[i] + (1.0f - cfg.beta1) * grad;
        m2_vec[i] = cfg.beta2 * m2_vec[i] + (1.0f - cfg.beta2) * grad * grad;

        float m1_hat = m1_vec[i] / bc1;
        float m2_term = m2_vec[i];
        (*vhat_vec)[i] = std::max((*vhat_vec)[i], m2_vec[i]);
        m2_term = (*vhat_vec)[i];
        float m2_hat = m2_term / bc2;
        data[i] -= lr * m1_hat / (std::sqrt(m2_hat) + cfg.epsilon);
      }

    } else {
      for (size_t i = 0; i < n; ++i) {
        float grad = grad_ptr[i];
        m1_vec[i] = cfg.beta1 * m1_vec[i] + (1.0f - cfg.beta1) * grad;
        m2_vec[i] = cfg.beta2 * m2_vec[i] + (1.0f - cfg.beta2) * grad * grad;

        float m1_hat = m1_vec[i] / bc1;
        float m2_term = m2_vec[i];
        float m2_hat = m2_term / bc2;
        data[i] -= lr * m1_hat / (std::sqrt(m2_hat) + cfg.epsilon);
      }
    }
  }
}

void adamw_step_packed_fused(AdamWPacked& packed, size_t& step_count,
                             const AdamWConfig& cfg) {
  const float lr = cfg.lr;
  ++step_count;
  const bool use_decay = cfg.use_weight_decay && cfg.weight_decay != 0.0f;
  const float beta1 = cfg.beta1;
  const float beta2 = cfg.beta2;
  packed.beta1_pow *= beta1;
  packed.beta2_pow *= beta2;
  const float bc1 = 1.0f - packed.beta1_pow;
  const float bc2 = 1.0f - packed.beta2_pow;
  const float one_minus_beta1 = 1.0f - beta1;
  const float one_minus_beta2 = 1.0f - beta2;
  const float inv_bc1 = 1.0f / bc1;
  const float inv_bc2 = 1.0f / bc2;
  const float weight_decay = cfg.weight_decay;
  const float epsilon = cfg.epsilon;

  const size_t total = packed.params * packed.elements;
  float* __restrict data = assume_aligned(packed.data.data());
  const float* __restrict grad = assume_aligned(packed.grad.data());
  float* __restrict m1 = assume_aligned(packed.m1.data());
  float* __restrict m2 = assume_aligned(packed.m2.data());
  float* vhat = cfg.amsgrad ? assume_aligned(packed.vhat.data()) : nullptr;

  if (cfg.amsgrad) {
    if (use_decay) {
      adamw_packed_update<true, true>(
          data, grad, m1, m2, vhat, total, beta1, beta2, one_minus_beta1,
          one_minus_beta2, inv_bc1, inv_bc2, weight_decay, lr, epsilon);
    } else {
      adamw_packed_update<false, true>(
          data, grad, m1, m2, vhat, total, beta1, beta2, one_minus_beta1,
          one_minus_beta2, inv_bc1, inv_bc2, weight_decay, lr, epsilon);
    }
  } else {
    if (use_decay) {
      adamw_packed_update<true, false>(
          data, grad, m1, m2, nullptr, total, beta1, beta2, one_minus_beta1,
          one_minus_beta2, inv_bc1, inv_bc2, weight_decay, lr, epsilon);
    } else {
      adamw_packed_update<false, false>(
          data, grad, m1, m2, nullptr, total, beta1, beta2, one_minus_beta1,
          one_minus_beta2, inv_bc1, inv_bc2, weight_decay, lr, epsilon);
    }
  }
}

double measure_adamw_ms(AdamWStep step, const AdamWConfig& cfg) {
  std::vector<Param> params = make_params(cfg);
  AdamWState opt_state = make_state(cfg);
  size_t step_count = 0;
  step(params, opt_state, step_count, cfg);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.iterations; ++i) {
    step(params, opt_state, step_count, cfg);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  benchmark::DoNotOptimize(params[0].data[0]);
  return elapsed.count() / static_cast<double>(cfg.iterations);
}

double measure_adamw_packed_ms(AdamWPackedStep step, const AdamWConfig& cfg) {
  AdamWPacked packed = make_packed_params(cfg);
  size_t step_count = 0;
  step(packed, step_count, cfg);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < cfg.iterations; ++i) {
    step(packed, step_count, cfg);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  benchmark::DoNotOptimize(packed.data[0]);
  return elapsed.count() / static_cast<double>(cfg.iterations);
}

void bench_adamw(benchmark::State& state, AdamWStep step,
                 const AdamWConfig& cfg, double multiplier) {
  std::vector<Param> params = make_params(cfg);
  AdamWState opt_state = make_state(cfg);
  size_t step_count = 0;
  step(params, opt_state, step_count, cfg);
  state.counters["speedup_vs_baseline"] = multiplier;
  for (auto _ : state) {
    step(params, opt_state, step_count, cfg);
    benchmark::DoNotOptimize(params[0].data[0]);
  }
}

void bench_adamw_packed_impl(benchmark::State& state, AdamWPackedStep step,
                             const AdamWConfig& cfg, double multiplier) {
  AdamWPacked packed = make_packed_params(cfg);
  size_t step_count = 0;
  step(packed, step_count, cfg);
  state.counters["speedup_vs_baseline"] = multiplier;
  for (auto _ : state) {
    step(packed, step_count, cfg);
    benchmark::DoNotOptimize(packed.data[0]);
  }
}

AdamWConfig parse_flags(int argc, char** argv) {
  AdamWConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("P=", 0) == 0) {
      cfg.params = std::atoi(arg.c_str() + 2);
    } else if (arg.rfind("N=", 0) == 0) {
      cfg.elements = std::atoi(arg.c_str() + 2);
    } else if (arg.rfind("I=", 0) == 0) {
      cfg.iterations = std::atoi(arg.c_str() + 2);
    } else if (arg.rfind("LR=", 0) == 0) {
      cfg.lr = std::strtof(arg.c_str() + 3, nullptr);
    } else if (arg.rfind("B1=", 0) == 0) {
      cfg.beta1 = std::strtof(arg.c_str() + 3, nullptr);
    } else if (arg.rfind("B2=", 0) == 0) {
      cfg.beta2 = std::strtof(arg.c_str() + 3, nullptr);
    } else if (arg.rfind("WD=", 0) == 0) {
      cfg.weight_decay = std::strtof(arg.c_str() + 3, nullptr);
    } else if (arg.rfind("USE_WD=", 0) == 0) {
      cfg.use_weight_decay = std::atoi(arg.c_str() + 7) != 0;
    } else if (arg.rfind("AMS=", 0) == 0) {
      cfg.amsgrad = std::atoi(arg.c_str() + 4) != 0;
    } else if (arg.rfind("EPS=", 0) == 0) {
      cfg.epsilon = std::strtof(arg.c_str() + 4, nullptr);
    } else if (arg.rfind("SEED=", 0) == 0) {
      cfg.seed =
          static_cast<uint32_t>(std::strtoul(arg.c_str() + 5, nullptr, 10));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(1);
    }
  }
  return cfg;
}

AdamWConfig g_cfg;
double g_baseline_ms = 0.0;
double g_packed_ms = 0.0;

void bench_adamw_baseline(benchmark::State& state) {
  bench_adamw(state, adamw_step_baseline, g_cfg, 1.0);
}

void bench_adamw_packed(benchmark::State& state) {
  double multiplier = 0.0;
  if (g_packed_ms > 0.0) {
    multiplier = g_baseline_ms / g_packed_ms;
  }
  bench_adamw_packed_impl(state, adamw_step_packed_fused, g_cfg, multiplier);
}

}  // namespace

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  g_cfg = parse_flags(argc, argv);
  if (g_cfg.params <= 0 || g_cfg.elements <= 0) {
    std::cerr << "P and N must be positive." << std::endl;
    return 1;
  }
  if (g_cfg.iterations <= 0) g_cfg.iterations = 1;

  g_baseline_ms = measure_adamw_ms(adamw_step_baseline, g_cfg);
  g_packed_ms = measure_adamw_packed_ms(adamw_step_packed_fused, g_cfg);

  auto* baseline =
      ::benchmark::RegisterBenchmark("adamw_baseline", bench_adamw_baseline);
  auto* packed =
      ::benchmark::RegisterBenchmark("adamw_packed", bench_adamw_packed);
  if (g_cfg.iterations > 0) {
    baseline->Iterations(g_cfg.iterations);
    packed->Iterations(g_cfg.iterations);
  }
  baseline->UseRealTime()->Unit(benchmark::kMillisecond);
  packed->UseRealTime()->Unit(benchmark::kMillisecond);

  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
