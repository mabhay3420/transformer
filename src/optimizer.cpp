#include "optimizer.hpp"
#include "micrograd.hpp"

void SGDOptimizer::zero_grad() {
  for (auto p : params) {
    // auto v = mem_pool->get(p);
    mem_pool->get(p)->grad = 0.0f;
  }
}

void SGDOptimizer::step() {
  auto CURR_LR = lr_scheduler.get();
  for (auto p : params) {
    mem_pool->get(p)->data += -CURR_LR * mem_pool->get(p)->grad;
  }
}