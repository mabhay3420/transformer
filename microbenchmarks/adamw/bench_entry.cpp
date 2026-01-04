#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <os/log.h>
#include <os/signpost.h>
#endif

#include "tensor.hpp"

class AdamWTest {
 public:
  AdamWTest(std::vector<Tensor> params, float beta1 = 0.9f,
            float beta2 = 0.999f, float weight_decay = 0.0f,
            bool use_weight_decay = false, bool amsgrad = false,
            float epsilon = 1e-8f)
      : beta1_(beta1),
        beta2_(beta2),
        weight_decay_(weight_decay),
        epsilon_(epsilon),
        params_(std::move(params)),
        step_count_(0),
        curr_lr(1.0f) {
    const size_t param_count = params_.size();
    m1_.resize(param_count);
    m2_.resize(param_count);
  }

  virtual ~AdamWTest() = default;

  virtual void step() = 0;

 protected:
  static void ensure_state_size(std::vector<float>& state, size_t target) {
    if (state.size() != target) {
      state.assign(target, 0.0f);
    }
  }

  static bool valid_param(const Tensor& t) {
    return t.numel > 0 && t.data() != nullptr && t.grad() != nullptr;
  }

  float beta1_;
  float beta2_;
  float weight_decay_;
  float epsilon_;
  std::vector<std::vector<float>> m1_;
  std::vector<std::vector<float>> m2_;
  std::vector<std::vector<float>> vhat_;
  std::vector<Tensor> params_;
  size_t step_count_;
  float curr_lr;
};

class AdamWTestBase : public AdamWTest {
 public:
  using AdamWTest::AdamWTest;

  void step() override;
};

void AdamWTestBase::step() {
  const float lr = curr_lr;
  ++step_count_;
  const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
  const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

  for (size_t idx = 0; idx < params_.size(); ++idx) {
    Tensor& param = params_[idx];
    if (!valid_param(param)) continue;
    float* data = param.data();
    const float* grad_ptr = param.grad();
    const size_t n = param.numel;
    ensure_state_size(m1_[idx], n);
    ensure_state_size(m2_[idx], n);
    auto& m1_vec = m1_[idx];
    auto& m2_vec = m2_[idx];
    std::vector<float>* vhat_vec = nullptr;

    for (size_t i = 0; i < n; ++i) {
      data[i] -= lr * weight_decay_ * data[i];
    }

    for (size_t i = 0; i < n; ++i) {
      float grad = grad_ptr[i];
      m1_vec[i] = beta1_ * m1_vec[i] + (1.0f - beta1_) * grad;
      m2_vec[i] = beta2_ * m2_vec[i] + (1.0f - beta2_) * grad * grad;
      float m1_hat = m1_vec[i] / bc1;
      float m2_hat = m2_vec[i] / bc2;
      data[i] -= lr * m1_hat / (std::sqrt(m2_hat) + epsilon_);
    }
  }
}

class AdamWTestInvBc : public AdamWTest {
 public:
  using AdamWTest::AdamWTest;

  void step() override;
};

void AdamWTestInvBc::step() {
  const float lr = curr_lr;
  ++step_count_;
  const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
  const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
  const float invbc1 = 1.0f / bc1;
  const float invbc2 = 1.0f / bc2;
  const float wd_scale = 1.0f - weight_decay_ * lr;

  for (size_t idx = 0; idx < params_.size(); ++idx) {
    Tensor& param = params_[idx];
    if (!valid_param(param)) continue;
    float* __restrict data = param.data();
    const float* __restrict grad_ptr = param.grad();
    const size_t n = param.numel;

    ensure_state_size(m1_[idx], n);
    ensure_state_size(m2_[idx], n);
    float* __restrict m1_vec = m1_[idx].data();
    float* __restrict m2_vec = m2_[idx].data();
    std::vector<float>* vhat_vec = nullptr;

    for (size_t i = 0; i < n; ++i) {
      float grad = grad_ptr[i];
      float m1 = m1_vec[i];
      float m2 = m2_vec[i];
      m1 = beta1_ * m1 + (1.0f - beta1_) * grad;
      m2 = beta2_ * m2 + (1.0f - beta2_) * grad * grad;
      m1_vec[i] = m1;
      m2_vec[i] = m2;
      float m1_hat = m1 * invbc1;
      float m2_hat = m2 * invbc2;
      float x = data[i];
      x *= wd_scale;
      x -= lr * m1_hat / (std::sqrt(m2_hat) + epsilon_);
      data[i] = x;
    }
  }
}

class AdamWTestInvBcCacheV1 : public AdamWTest {
 public:
  using AdamWTest::AdamWTest;

  void step() override;
};
void AdamWTestInvBcCacheV1::step() {
  const float lr = curr_lr;
  ++step_count_;
  const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
  const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
  const float invbc1 = 1.0f / bc1;
  const float invbc2 = 1.0f / bc2;
  const float wd_scale = 1.0f - weight_decay_ * lr;

  for (size_t idx = 0; idx < params_.size(); ++idx) {
    Tensor& param = params_[idx];
    if (!valid_param(param)) continue;
    float* data = param.data();
    const float* grad_ptr = param.grad();
    const size_t n = param.numel;
    ensure_state_size(m1_[idx], n);
    ensure_state_size(m2_[idx], n);
    auto& m1_vec = m1_[idx];
    auto& m2_vec = m2_[idx];
    std::vector<float>* vhat_vec = nullptr;

    const int B = 1024;
    for (size_t base = 0; base < n; base += B) {
      const size_t end = std::min(n, base + B);
      for (size_t i = base; i < end; ++i) {
        float grad = grad_ptr[i];
        float m1 = m1_vec[i];
        float m2 = m2_vec[i];
        m1 = beta1_ * m1 + (1.0f - beta1_) * grad;
        m2 = beta2_ * m2 + (1.0f - beta2_) * grad * grad;
        m1_vec[i] = m1;
        m2_vec[i] = m2;
        float m1_hat = m1 * invbc1;
        float m2_hat = m2 * invbc2;
        float x = data[i];
        x *= wd_scale;
        x -= lr * m1_hat / (std::sqrt(m2_hat) + epsilon_);
        data[i] = x;
      }
    }
  }
}

class AdamWTestInvBcVector1 : public AdamWTest {
 public:
  using AdamWTest::AdamWTest;

  void step() override;
};
void AdamWTestInvBcVector1::step() {
  const float lr = curr_lr;
  ++step_count_;
  const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
  const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
  const float invbc1 = 1.0f / bc1;
  const float invbc2 = 1.0f / bc2;
  const float wd_scale = 1.0f - weight_decay_ * lr;

  for (size_t idx = 0; idx < params_.size(); ++idx) {
    Tensor& param = params_[idx];
    if (!valid_param(param)) continue;
    float* __restrict data = param.data();
    const float* __restrict grad_ptr = param.grad();
    const size_t n = param.numel;
    ensure_state_size(m1_[idx], n);
    ensure_state_size(m2_[idx], n);
    float* __restrict m1_vec = m1_[idx].data();
    float* __restrict m2_vec = m2_[idx].data();
    std::vector<float>* vhat_vec = nullptr;

    for (size_t i = 0; i < n; ++i) {
      float grad = grad_ptr[i];
      float m1 = m1_vec[i];
      float m2 = m2_vec[i];
      m1 = beta1_ * m1 + (1.0f - beta1_) * grad;
      m2 = beta2_ * m2 + (1.0f - beta2_) * grad * grad;
      m1_vec[i] = m1;
      m2_vec[i] = m2;
      float m1_hat = m1 * invbc1;
      float m2_hat = m2 * invbc2;
      float x = data[i];
      x *= wd_scale;
      x -= lr * m1_hat / (std::sqrt(m2_hat) + epsilon_);
      data[i] = x;
    }
  }
}

namespace {

#if defined(__APPLE__)
os_log_t benchmark_log() {
  static os_log_t log =
      os_log_create("com.transformer.bench", "PointsOfInterest");
  return log;
}
#endif

constexpr float kBeta1 = 0.9f;
constexpr float kBeta2 = 0.999f;
constexpr float kWeightDecay = 0.01f;
constexpr float kEpsilon = 1e-8f;
constexpr size_t kParamCount = 4;

std::vector<Tensor> make_params(ParameterStore& store, size_t numel) {
  store.reserve(numel * kParamCount);
  std::vector<Tensor> params;
  params.reserve(kParamCount);
  const int dim = static_cast<int>(numel);
  for (size_t p = 0; p < kParamCount; ++p) {
    Tensor param = store.tensor({dim}, TensorInit::UninitializedData);
    float* data = param.data();
    float* grad = param.grad();
    for (size_t i = 0; i < numel; ++i) {
      const int data_val = static_cast<int>((i + p) % 97) - 48;
      const int grad_val = static_cast<int>((i + 2 * p) % 53) - 26;
      data[i] = static_cast<float>(data_val) * 0.01f;
      grad[i] = static_cast<float>(grad_val) * 0.01f;
    }
    params.push_back(param);
  }
  return params;
}

template <typename AdamWImpl>
void run_bench(benchmark::State& state) {
  const size_t numel = static_cast<size_t>(state.range(0));
  ParameterStore store;
  auto params = make_params(store, numel);
  AdamWImpl optimizer(std::move(params), kBeta1, kBeta2, kWeightDecay, true,
                      false, kEpsilon);
  optimizer.step();
  for (auto _ : state) {
    optimizer.step();
  }
  const int64_t elems = static_cast<int64_t>(state.iterations()) *
                        static_cast<int64_t>(kParamCount) *
                        static_cast<int64_t>(numel);
  state.SetItemsProcessed(elems);
}

void BM_AdamWBase(benchmark::State& state) {
#if defined(__APPLE__)
  os_log_t log = benchmark_log();
#endif
#if defined(__APPLE__)
  os_signpost_interval_begin(log, OS_SIGNPOST_ID_EXCLUSIVE, "adamw_step");
#endif
  run_bench<AdamWTestBase>(state);
#if defined(__APPLE__)
  os_signpost_interval_end(log, OS_SIGNPOST_ID_EXCLUSIVE, "adamw_step");
#endif
}

void BM_AdamWInvBc(benchmark::State& state) {
#if defined(__APPLE__)
  os_log_t log = benchmark_log();
#endif
#if defined(__APPLE__)
  os_signpost_interval_begin(log, OS_SIGNPOST_ID_EXCLUSIVE,
                             "adamw_step_inv_bc");
#endif
  run_bench<AdamWTestInvBc>(state);
#if defined(__APPLE__)
  os_signpost_interval_end(log, OS_SIGNPOST_ID_EXCLUSIVE, "adamw_step_inv_bc");
#endif
}

void BM_AdamWInvBcCacheV1(benchmark::State& state) {
#if defined(__APPLE__)
  os_log_t log = benchmark_log();
#endif
#if defined(__APPLE__)
  os_signpost_interval_begin(log, OS_SIGNPOST_ID_EXCLUSIVE,
                             "adamw_step_inv_bc_cache_v1");
#endif
  run_bench<AdamWTestInvBc>(state);
#if defined(__APPLE__)
  os_signpost_interval_end(log, OS_SIGNPOST_ID_EXCLUSIVE,
                           "adamw_step_inv_bc_cache_v1");
#endif
}
}  // namespace

const int multiplier = 4;
// const int range_start = 1 << 10;
const int range_start = 1 << 16;
const int range_end = 1 << 16;
// BENCHMARK(BM_AdamWBase)->RangeMultiplier(4)->Range(1 << 10, 1 << 16);
BENCHMARK(BM_AdamWInvBc)
    ->RangeMultiplier(multiplier)
    ->Range(range_start, range_end);

// BENCHMARK(BM_AdamWInvBcVector1)
//     ->RangeMultiplier(multiplier)
//     ->Range(range_start, range_end);
BENCHMARK_MAIN();
