#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "tensor.hpp"

namespace optim {

class Optimizer {
 protected:
  std::vector<Tensor> params_;
  size_t step_count_;

 public:
  explicit Optimizer(std::vector<Tensor> params)
      : params_(std::move(params)), step_count_(0) {}
  virtual ~Optimizer() = default;

  void zero_grad() {
    for (auto &param : params_) {
      param.zero_grad();
    }
  }

  virtual void step() = 0;

 protected:
  static void ensure_state_size(std::vector<float> &state, size_t target) {
    if (state.size() != target) {
      state.assign(target, 0.0f);
    }
  }

  static bool valid_param(const Tensor &t) {
    return t.numel > 0 && t.data() != nullptr && t.grad() != nullptr;
  }
};

template <typename Scheduler>
class OptimizerWithScheduler : public Optimizer {
 protected:
  Scheduler *scheduler_;

 public:
  OptimizerWithScheduler(std::vector<Tensor> params, Scheduler &scheduler)
      : Optimizer(std::move(params)), scheduler_(&scheduler) {}
};

template <typename Scheduler>
class SGD : public OptimizerWithScheduler<Scheduler> {
 public:
  SGD(std::vector<Tensor> params, Scheduler &scheduler,
      float momentum_beta = 0.0f)
      : OptimizerWithScheduler<Scheduler>(std::move(params), scheduler),
        momentum_beta_(momentum_beta),
        momentum_(this->params_.size()) {}

  void step() override {
    const float lr = this->scheduler_->get();
    ++this->step_count_;

    for (size_t idx = 0; idx < this->params_.size(); ++idx) {
      Tensor &param = this->params_[idx];
      if (!Optimizer::valid_param(param)) continue;
      float *data = param.data();
      const float *grad = param.grad();
      const size_t n = param.numel;
      if (momentum_beta_ != 0.0f) {
        Optimizer::ensure_state_size(momentum_[idx], n);
        auto &momentum_vec = momentum_[idx];
        for (size_t i = 0; i < n; ++i) {
          momentum_vec[i] = momentum_beta_ * momentum_vec[i] + grad[i];
          data[i] -= lr * momentum_vec[i];
        }
      } else {
        for (size_t i = 0; i < n; ++i) {
          data[i] -= lr * grad[i];
        }
      }
    }
  }

 private:
  float momentum_beta_;
  std::vector<std::vector<float>> momentum_;
};

template <typename Scheduler>
class Adam : public OptimizerWithScheduler<Scheduler> {
 public:
  Adam(std::vector<Tensor> params, Scheduler &scheduler, float beta1 = 0.9f,
       float beta2 = 0.999f, float weight_decay = 0.0f, bool amsgrad = false,
       float epsilon = 1e-8f)
      : OptimizerWithScheduler<Scheduler>(std::move(params), scheduler),
        beta1_(beta1),
        beta2_(beta2),
        weight_decay_(weight_decay),
        amsgrad_(amsgrad),
        epsilon_(epsilon),
        m1_(this->params_.size()),
        m2_(this->params_.size()),
        vhat_(amsgrad ? this->params_.size() : 0) {}

  void step() override {
    const float lr = this->scheduler_->get();
    ++this->step_count_;
    const float bc1 =
        1.0f - std::pow(beta1_, static_cast<float>(this->step_count_));
    const float bc2 =
        1.0f - std::pow(beta2_, static_cast<float>(this->step_count_));

    for (size_t idx = 0; idx < this->params_.size(); ++idx) {
      Tensor &param = this->params_[idx];
      if (!Optimizer::valid_param(param)) continue;
      float *data = param.data();
      const float *grad_ptr = param.grad();
      const size_t n = param.numel;
      Optimizer::ensure_state_size(m1_[idx], n);
      Optimizer::ensure_state_size(m2_[idx], n);
      if (amsgrad_) {
        Optimizer::ensure_state_size(vhat_[idx], n);
      }
      auto &m1_vec = m1_[idx];
      auto &m2_vec = m2_[idx];
      std::vector<float> *vhat_vec = amsgrad_ ? &vhat_[idx] : nullptr;

      for (size_t i = 0; i < n; ++i) {
        float grad = grad_ptr[i];
        if (weight_decay_ != 0.0f) {
          grad += weight_decay_ * data[i];
        }

        m1_vec[i] = beta1_ * m1_vec[i] + (1.0f - beta1_) * grad;
        m2_vec[i] = beta2_ * m2_vec[i] + (1.0f - beta2_) * grad * grad;

        float m1_hat = m1_vec[i] / bc1;
        float m2_term = m2_vec[i];
        if (amsgrad_) {
          (*vhat_vec)[i] = std::max((*vhat_vec)[i], m2_vec[i]);
          m2_term = (*vhat_vec)[i];
        }
        float m2_hat = m2_term / bc2;
        data[i] -= lr * m1_hat / (std::sqrt(m2_hat) + epsilon_);
      }
    }
  }

 private:
  float beta1_;
  float beta2_;
  float weight_decay_;
  bool amsgrad_;
  float epsilon_;
  std::vector<std::vector<float>> m1_;
  std::vector<std::vector<float>> m2_;
  std::vector<std::vector<float>> vhat_;
};

template <typename Scheduler>
class AdamW : public OptimizerWithScheduler<Scheduler> {
 public:
  AdamW(std::vector<Tensor> params, Scheduler &scheduler, float beta1 = 0.9f,
        float beta2 = 0.999f, float weight_decay = 0.0f, bool amsgrad = false,
        float epsilon = 1e-8f)
      : OptimizerWithScheduler<Scheduler>(std::move(params), scheduler),
        beta1_(beta1),
        beta2_(beta2),
        weight_decay_(weight_decay),
        amsgrad_(amsgrad),
        epsilon_(epsilon),
        m1_(this->params_.size()),
        m2_(this->params_.size()),
        vhat_(amsgrad ? this->params_.size() : 0) {}

  void step() override {
    const float lr = this->scheduler_->get();
    ++this->step_count_;
    const float bc1 =
        1.0f - std::pow(beta1_, static_cast<float>(this->step_count_));
    const float bc2 =
        1.0f - std::pow(beta2_, static_cast<float>(this->step_count_));

    for (size_t idx = 0; idx < this->params_.size(); ++idx) {
      Tensor &param = this->params_[idx];
      if (!Optimizer::valid_param(param)) continue;
      float *data = param.data();
      const float *grad_ptr = param.grad();
      const size_t n = param.numel;
      Optimizer::ensure_state_size(m1_[idx], n);
      Optimizer::ensure_state_size(m2_[idx], n);
      if (amsgrad_) {
        Optimizer::ensure_state_size(vhat_[idx], n);
      }
      auto &m1_vec = m1_[idx];
      auto &m2_vec = m2_[idx];
      std::vector<float> *vhat_vec = amsgrad_ ? &vhat_[idx] : nullptr;

      if (weight_decay_ != 0.0f) {
        for (size_t i = 0; i < n; ++i) {
          data[i] -= lr * weight_decay_ * data[i];
        }
      }

      for (size_t i = 0; i < n; ++i) {
        float grad = grad_ptr[i];
        m1_vec[i] = beta1_ * m1_vec[i] + (1.0f - beta1_) * grad;
        m2_vec[i] = beta2_ * m2_vec[i] + (1.0f - beta2_) * grad * grad;

        float m1_hat = m1_vec[i] / bc1;
        float m2_term = m2_vec[i];
        if (amsgrad_) {
          (*vhat_vec)[i] = std::max((*vhat_vec)[i], m2_vec[i]);
          m2_term = (*vhat_vec)[i];
        }
        float m2_hat = m2_term / bc2;
        data[i] -= lr * m1_hat / (std::sqrt(m2_hat) + epsilon_);
      }
    }
  }

 private:
  float beta1_;
  float beta2_;
  float weight_decay_;
  bool amsgrad_;
  float epsilon_;
  std::vector<std::vector<float>> m1_;
  std::vector<std::vector<float>> m2_;
  std::vector<std::vector<float>> vhat_;
};

}  // namespace optim
