#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensor.hpp"

namespace nn {

struct Module {
  virtual ~Module() = default;
  virtual Tensor forward(const Tensor &x, ParameterStore &store) = 0;
  Tensor operator()(const Tensor &x, ParameterStore &store) {
    return forward(x, store);
  }
  virtual std::vector<Tensor> params() { return {}; }
};

struct Linear : public Module {
  int in_features;
  int out_features;
  bool use_bias;
  Tensor W;
  Tensor b;  // if use_bias==false, b.numel==0

  Linear(int in_f, int out_f, ParameterStore &store, bool bias = true,
         float init_scale = 0.5f, unsigned seed = 0);
  Tensor forward(const Tensor &x, ParameterStore &store) override;
  std::vector<Tensor> params() override;
};

struct Tanh : public Module {
  Tensor forward(const Tensor &x, ParameterStore &store) override;
};

struct Relu : public Module {
  Tensor forward(const Tensor &x, ParameterStore &store) override;
};

struct Sigmoid : public Module {
  Tensor forward(const Tensor &x, ParameterStore &store) override;
};

struct Sequential : public Module {
  std::vector<std::unique_ptr<Module>> layers;
  Sequential() = default;
  Tensor forward(const Tensor &x, ParameterStore &store) override;
  std::vector<Tensor> params() override;
  void push_back(std::unique_ptr<Module> m) { layers.push_back(std::move(m)); }
  template <typename T, typename... Args>
  T &emplace_back(Args &&...args) {
    static_assert(std::is_base_of_v<Module, T>,
                  "Sequential accepts Module-derived layers only");
    auto layer = std::make_unique<T>(std::forward<Args>(args)...);
    T &ref = *layer;
    layers.push_back(std::move(layer));
    return ref;
  }
};

// Losses
Tensor bce_with_logits_loss(const Tensor &logits, const Tensor &targets,
                            ParameterStore &store, float eps = 1e-6f);

// Optim
inline void sgd_step(const std::vector<Tensor> &params, float lr) {
  for (const auto &p : params) {
    float *w = p.store ? p.store->data_ptr(p.offset) : nullptr;
    float *g = p.store ? p.store->grad_ptr(p.offset) : nullptr;
    if (!w || !g) continue;
    for (size_t i = 0; i < p.numel; ++i) {
      w[i] -= lr * g[i];
    }
  }
}

}  // namespace nn
