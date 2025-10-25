/**
 * @file optimizer.hpp
 * @brief Optimization algorithms for training neural networks.
 *
 * Provides SGD, Adam, and AdamW optimizers with configurable learning rate
 * schedulers. Supports momentum, weight decay, and AMSGrad variants.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "tensor.hpp"
#include "utils.hpp"

namespace optim {

/**
 * @class Optimizer
 * @brief Base class for optimization algorithms.
 *
 * Manages parameter updates and gradient zeroing. Subclasses implement
 * specific update rules.
 */
class Optimizer {
 protected:
  std::vector<Tensor> params_;  ///< Learnable parameters to optimize
  size_t step_count_;           ///< Number of steps taken

 public:
  /**
   * @brief Construct optimizer with parameters.
   * @param params List of parameter tensors
   */
  explicit Optimizer(std::vector<Tensor> params)
      : params_(std::move(params)), step_count_(0) {}
  virtual ~Optimizer() = default;

  /**
   * @brief Zero gradients for all parameters.
   */
  void zero_grad() {
    for (auto& param : params_) {
      param.zero_grad();
    }
  }

  /**
   * @brief Perform one optimization step.
   */
  virtual void step() = 0;

 protected:
  /**
   * @brief Ensure optimizer state vector has correct size.
   * @param state State vector to resize
   * @param target Target size
   */
  static void ensure_state_size(std::vector<float>& state, size_t target) {
    if (state.size() != target) {
      state.assign(target, 0.0f);
    }
  }

  /**
   * @brief Check if parameter tensor is valid for optimization.
   * @param t Parameter tensor
   * @return True if valid
   */
  static bool valid_param(const Tensor& t) {
    return t.numel > 0 && t.data() != nullptr && t.grad() != nullptr;
  }
};

/**
 * @class OptimizerWithScheduler
 * @brief Base class for optimizers that use learning rate schedulers.
 * @tparam Scheduler Learning rate scheduler type
 */
template <typename Scheduler>
class OptimizerWithScheduler : public Optimizer {
 protected:
  Scheduler* scheduler_;  ///< Pointer to learning rate scheduler

 public:
  /**
   * @brief Construct optimizer with scheduler.
   * @param params Parameter tensors
   * @param scheduler Learning rate scheduler reference
   */
  OptimizerWithScheduler(std::vector<Tensor> params, Scheduler& scheduler)
      : Optimizer(std::move(params)), scheduler_(&scheduler) {}
};

/**
 * @class SGD
 * @brief Stochastic Gradient Descent optimizer with momentum.
 * @tparam Scheduler Learning rate scheduler type
 *
 * Implements SGD with optional momentum: v = βv + ∇, p -= lr * v
 */
template <typename Scheduler>
class SGD : public OptimizerWithScheduler<Scheduler> {
 public:
  /**
   * @brief Construct SGD optimizer.
   * @param params Parameter tensors
   * @param scheduler Learning rate scheduler
   * @param momentum_beta Momentum coefficient (0 for no momentum)
   */
  SGD(std::vector<Tensor> params, Scheduler& scheduler,
      float momentum_beta = 0.0f)
      : OptimizerWithScheduler<Scheduler>(std::move(params), scheduler),
        momentum_beta_(momentum_beta),
        momentum_(this->params_.size()) {}

  /**
   * @brief Perform SGD update step.
   */
  void step() override {
    const float lr = this->scheduler_->get();
    ++this->step_count_;

    for (size_t idx = 0; idx < this->params_.size(); ++idx) {
      Tensor& param = this->params_[idx];
      if (!Optimizer::valid_param(param)) continue;
      float* data = param.data();
      const float* grad = param.grad();
      const size_t n = param.numel;
      if (momentum_beta_ != 0.0f) {
        Optimizer::ensure_state_size(momentum_[idx], n);
        auto& momentum_vec = momentum_[idx];
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

/**
 * @class Adam
 * @brief Adam optimizer with weight decay.
 * @tparam Scheduler Learning rate scheduler type
 *
 * Implements Adam algorithm: m = β1*m + (1-β1)*∇, v = β2*v + (1-β2)*∇²
 * p -= lr * m̂ / (√v̂ + ε), with optional AMSGrad and weight decay.
 */
template <typename Scheduler>
class Adam : public OptimizerWithScheduler<Scheduler> {
 public:
  /**
   * @brief Construct Adam optimizer.
   * @param params Parameter tensors
   * @param scheduler Learning rate scheduler
   * @param beta1 First moment decay rate (default 0.9)
   * @param beta2 Second moment decay rate (default 0.999)
   * @param weight_decay Weight decay coefficient (default 0.0)
   * @param use_weight_decay Use weight decay (default false)
   * @param amsgrad Use AMSGrad variant (default false)
   * @param epsilon Numerical stability constant (default 1e-8)
   */
  Adam(std::vector<Tensor> params, Scheduler& scheduler, float beta1 = 0.9f,
       float beta2 = 0.999f, float weight_decay = 0.0f,
       bool use_weight_decay = false, bool amsgrad = false,
       float epsilon = 1e-8f)
      : OptimizerWithScheduler<Scheduler>(std::move(params), scheduler),
        beta1_(beta1),
        beta2_(beta2),
        weight_decay_(weight_decay),
        use_weight_decay_(use_weight_decay),
        amsgrad_(amsgrad),
        epsilon_(epsilon),
        m1_(this->params_.size()),
        m2_(this->params_.size()),
        vhat_(amsgrad ? this->params_.size() : 0) {}

  /**
   * @brief Perform Adam update step.
   */
  void step() override {
    const float lr = this->scheduler_->get();
    ++this->step_count_;
    const float bc1 =
        1.0f - std::pow(beta1_, static_cast<float>(this->step_count_));
    const float bc2 =
        1.0f - std::pow(beta2_, static_cast<float>(this->step_count_));

    for (size_t idx = 0; idx < this->params_.size(); ++idx) {
      Tensor& param = this->params_[idx];
      if (!Optimizer::valid_param(param)) continue;
      float* data = param.data();
      const float* grad_ptr = param.grad();
      const size_t n = param.numel;
      Optimizer::ensure_state_size(m1_[idx], n);
      Optimizer::ensure_state_size(m2_[idx], n);
      if (amsgrad_) {
        Optimizer::ensure_state_size(vhat_[idx], n);
      }
      auto& m1_vec = m1_[idx];
      auto& m2_vec = m2_[idx];
      std::vector<float>* vhat_vec = amsgrad_ ? &vhat_[idx] : nullptr;

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
  bool use_weight_decay_;
  bool amsgrad_;
  float epsilon_;
  std::vector<std::vector<float>> m1_;
  std::vector<std::vector<float>> m2_;
  std::vector<std::vector<float>> vhat_;
};

/**
 * @class AdamW
 * @brief AdamW optimizer with decoupled weight decay.
 * @tparam Scheduler Learning rate scheduler type
 *
 * Similar to Adam but applies weight decay before gradient update:
 * p -= lr * λ * p, then Adam update on the decayed gradient.
 */
template <typename Scheduler>
class AdamW : public OptimizerWithScheduler<Scheduler> {
 public:
  /**
   * @brief Construct AdamW optimizer.
   * @param params Parameter tensors
   * @param scheduler Learning rate scheduler
   * @param beta1 First moment decay rate (default 0.9)
   * @param beta2 Second moment decay rate (default 0.999)
   * @param weight_decay Weight decay coefficient (default 0.0)
   * @param use_weight_decay Use weight decay (default false)
   * @param amsgrad Use AMSGrad variant (default false)
   * @param epsilon Numerical stability constant (default 1e-8)
   */
  AdamW(std::vector<Tensor> params, Scheduler& scheduler, float beta1 = 0.9f,
        float beta2 = 0.999f, float weight_decay = 0.0f,
        bool use_weight_decay = false, bool amsgrad = false,
        float epsilon = 1e-8f)
      : OptimizerWithScheduler<Scheduler>(std::move(params), scheduler),
        beta1_(beta1),
        beta2_(beta2),
        weight_decay_(weight_decay),
        use_weight_decay_(use_weight_decay),
        amsgrad_(amsgrad),
        epsilon_(epsilon),
        m1_(this->params_.size()),
        m2_(this->params_.size()),
        vhat_(amsgrad ? this->params_.size() : 0) {}

  /**
   * @brief Perform AdamW update step.
   */
  void step() override {
    const float lr = this->scheduler_->get();
    ++this->step_count_;
    const float bc1 =
        1.0f - std::pow(beta1_, static_cast<float>(this->step_count_));
    const float bc2 =
        1.0f - std::pow(beta2_, static_cast<float>(this->step_count_));

    for (size_t idx = 0; idx < this->params_.size(); ++idx) {
      Tensor& param = this->params_[idx];
      if (!Optimizer::valid_param(param)) continue;
      float* data = param.data();
      const float* grad_ptr = param.grad();
      const size_t n = param.numel;
      Optimizer::ensure_state_size(m1_[idx], n);
      Optimizer::ensure_state_size(m2_[idx], n);
      if (amsgrad_) {
        Optimizer::ensure_state_size(vhat_[idx], n);
      }
      auto& m1_vec = m1_[idx];
      auto& m2_vec = m2_[idx];
      std::vector<float>* vhat_vec = amsgrad_ ? &vhat_[idx] : nullptr;

      if (use_weight_decay_) {
        for (size_t i = 0; i < n; ++i) {
          data[i] -= lr * weight_decay_ * data[i];
        }
      }

      if (amsgrad_) {
        for (size_t i = 0; i < n; ++i) {
          float grad = grad_ptr[i];
          m1_vec[i] = beta1_ * m1_vec[i] + (1.0f - beta1_) * grad;
          m2_vec[i] = beta2_ * m2_vec[i] + (1.0f - beta2_) * grad * grad;

          // 55%
          float m1_hat = m1_vec[i] / bc1;
          float m2_term = m2_vec[i];
          (*vhat_vec)[i] = std::max((*vhat_vec)[i], m2_vec[i]);
          m2_term = (*vhat_vec)[i];
          float m2_hat = m2_term / bc2;
          // 30%
          data[i] -= lr * m1_hat / (std::sqrt(m2_hat) + epsilon_);
        }

      } else {
        for (size_t i = 0; i < n; ++i) {
          float grad = grad_ptr[i];
          m1_vec[i] = beta1_ * m1_vec[i] + (1.0f - beta1_) * grad;
          m2_vec[i] = beta2_ * m2_vec[i] + (1.0f - beta2_) * grad * grad;

          // 55%
          float m1_hat = m1_vec[i] / bc1;
          float m2_term = m2_vec[i];
          float m2_hat = m2_term / bc2;
          // 30%
          data[i] -= lr * m1_hat / (std::sqrt(m2_hat) + epsilon_);
        }
      }
    }
  }

 private:
  float beta1_;
  float beta2_;
  float weight_decay_;
  bool use_weight_decay_;
  bool amsgrad_;
  float epsilon_;
  std::vector<std::vector<float>> m1_;
  std::vector<std::vector<float>> m2_;
  std::vector<std::vector<float>> vhat_;
};

}  // namespace optim
