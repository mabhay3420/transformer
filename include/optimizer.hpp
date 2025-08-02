#pragma once

#include "learning_rate.hpp"
#include "mempool.hpp"
#include <memory>
template <typename T, typename LRScheduleType> struct Optimizer {
  std::shared_ptr<MemPool<Value>> mem_pool;
  std::vector<MemPoolIndex> params;
  LRScheduler<LRScheduleType> lr_scheduler;

  Optimizer(std::shared_ptr<MemPool<Value>> mem_pool,
            std::vector<MemPoolIndex> params,
            LRScheduler<LRScheduleType> lr_scheduler)
      : mem_pool(mem_pool), params(params), lr_scheduler(lr_scheduler){};

  void zero_grad() { static_cast<T *>(this)->zero_grad(); }
  void step() { static_cast<T *>(this)->step(); }
};

struct SGDOptimizer : Optimizer<SGDOptimizer, StepLRScheduler> {
  SGDOptimizer(std::shared_ptr<MemPool<Value>> mem_pool,
               std::vector<MemPoolIndex> params, StepLRScheduler lr_scheduler)
      : Optimizer<SGDOptimizer, StepLRScheduler>(mem_pool, params,
                                                 lr_scheduler){};

  void zero_grad();
  void step();
};