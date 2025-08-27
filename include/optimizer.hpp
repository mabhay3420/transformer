#pragma once

#include <memory>
#include <unordered_map>

#include "mempool.hpp"
template <typename T>
struct Optimizer {
  int step_count;
  MemPool<Value> *mem_pool;
  std::vector<MemPoolIndex> params;

  Optimizer(MemPool<Value> *mem_pool, std::vector<MemPoolIndex> params,
            float momentum_beta = 0.0f)
      : mem_pool(mem_pool), params(params), step_count(0) {}

  void zero_grad() {
    for (auto p : params) {
      mem_pool->get(p)->grad = 0.0f;
    }
  }
  void step() { static_cast<T *>(this)->step(); }
};

template <typename LRScheduleType>
struct OptimizerWithLRSchedule
    : Optimizer<OptimizerWithLRSchedule<LRScheduleType>> {
  using Base = Optimizer<OptimizerWithLRSchedule<LRScheduleType>>;
  using Base::mem_pool;
  using Base::params;
  using Base::step_count;
  LRScheduleType &lr_scheduler;
  OptimizerWithLRSchedule(MemPool<Value> *mem_pool,
                          std::vector<MemPoolIndex> params,
                          LRScheduleType &lr_scheduler)
      : Optimizer<OptimizerWithLRSchedule<LRScheduleType>>(mem_pool, params),
        lr_scheduler(lr_scheduler) {}
  void step();
};

template <typename LRSchedulerType>
struct SGDOptimizer : OptimizerWithLRSchedule<LRSchedulerType> {
  using Base = OptimizerWithLRSchedule<LRSchedulerType>;
  using Base::lr_scheduler;
  using Base::mem_pool;
  using Base::params;
  using Base::step_count;
  float momentum_beta;
  std::unordered_map<MemPoolIndex, float> momentum;
  SGDOptimizer(MemPool<Value> *mem_pool, std::vector<MemPoolIndex> params,
               LRSchedulerType &lr_scheduler, float momentum_beta = 0.0f)
      : Base(mem_pool, params, lr_scheduler), momentum_beta(momentum_beta) {}
  void step() {
    step_count++;
    auto CURR_LR = lr_scheduler.get();
    for (auto p : params) {
      momentum[p] = (momentum_beta * momentum[p]) + mem_pool->get(p)->grad;
      mem_pool->get(p)->data += -CURR_LR * momentum[p];
    }
  }
};

template <typename LRSchedulerType>
struct AdamOptimizer : OptimizerWithLRSchedule<LRSchedulerType> {
  using Base = OptimizerWithLRSchedule<LRSchedulerType>;
  using Base::lr_scheduler;
  using Base::mem_pool;
  using Base::params;
  using Base::step_count;
  float beta1;
  float beta2;
  float weight_decay;
  bool amsgrad;
  float epsilon;
  std::unordered_map<MemPoolIndex, float> moment_1;
  std::unordered_map<MemPoolIndex, float> moment_2;
  std::unordered_map<MemPoolIndex, float> moment_2_max;

  AdamOptimizer(MemPool<Value> *mem_pool, std::vector<MemPoolIndex> params,
                LRSchedulerType &lr_scheduler, float beta1 = 0.9f,
                float beta2 = 0.999f, float weight_decay = 0,
                bool amsgrad = false, float epsilon = 1e-8f)
      : OptimizerWithLRSchedule<LRSchedulerType>(mem_pool, params,
                                                 lr_scheduler),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        amsgrad(amsgrad),
        epsilon(epsilon){};

  void step() {
    step_count++;
    auto CURR_LR = lr_scheduler.get();
    for (auto p : params) {
      if (weight_decay > 0.0f) {
        mem_pool->get(p)->grad += weight_decay * mem_pool->get(p)->data;
      }
      moment_1[p] =
          (beta1 * moment_1[p]) + (1.0f - beta1) * mem_pool->get(p)->grad;
      moment_2[p] = (beta2 * moment_2[p]) + (1.0f - beta2) *
                                                mem_pool->get(p)->grad *
                                                mem_pool->get(p)->grad;
      auto moment_1_norm = moment_1[p] / (1.0f - std::pow(beta1, step_count));
      float moment_2_norm;
      if (amsgrad) {
        moment_2_max[p] = std::max(moment_2_max[p], moment_2[p]);
        moment_2_norm = moment_2_max[p] / (1.0f - std::pow(beta2, step_count));
      } else {
        moment_2_norm = moment_2[p] / (1.0f - std::pow(beta2, step_count));
      }
      mem_pool->get(p)->data +=
          -CURR_LR * moment_1_norm / (std::sqrt(moment_2_norm) + epsilon);
    }
  }
};

template <typename LRSchedulerType>
struct AdamWOptimizer : OptimizerWithLRSchedule<LRSchedulerType> {
  using Base = OptimizerWithLRSchedule<LRSchedulerType>;
  using Base::lr_scheduler;
  using Base::mem_pool;
  using Base::params;
  using Base::step_count;
  float beta1;
  float beta2;
  float weight_decay;
  bool amsgrad;
  float epsilon;
  std::unordered_map<MemPoolIndex, float> moment_1;
  std::unordered_map<MemPoolIndex, float> moment_2;
  std::unordered_map<MemPoolIndex, float> moment_2_max;
  std::unordered_map<MemPoolIndex, float> moment_1_max;

  AdamWOptimizer(MemPool<Value> *mem_pool, std::vector<MemPoolIndex> params,
                 LRSchedulerType &lr_scheduler, float beta1 = 0.9f,
                 float beta2 = 0.999f, float weight_decay = 0,
                 bool amsgrad = false, float epsilon = 1e-8f)
      : OptimizerWithLRSchedule<LRSchedulerType>(mem_pool, params,
                                                 lr_scheduler),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        amsgrad(amsgrad),
        epsilon(epsilon){};

  void step() {
    step_count++;
    auto CURR_LR = lr_scheduler.get();
    for (auto p : params) {
      if (weight_decay > 0.0f) {
        mem_pool->get(p)->data -=
            weight_decay * CURR_LR * mem_pool->get(p)->data;
      }

      moment_1[p] =
          beta1 * moment_1[p] + (1.0f - beta1) * mem_pool->get(p)->grad;
      moment_2[p] = beta2 * moment_2[p] + (1.0f - beta2) *
                                              mem_pool->get(p)->grad *
                                              mem_pool->get(p)->grad;
      auto moment_1_norm = moment_1[p] / (1.0f - std::pow(beta1, step_count));

      float moment_2_norm;
      if (amsgrad) {
        moment_2_max[p] = std::max(moment_2_max[p], moment_2[p]);
        moment_2_norm = moment_2_max[p] / (1.0f - std::pow(beta2, step_count));
      } else {
        moment_2_norm = moment_2[p] / (1.0f - std::pow(beta2, step_count));
      }
      mem_pool->get(p)->data -=
          CURR_LR * moment_1_norm / ((std::sqrt(moment_2_norm) + epsilon));
    }
  }
};