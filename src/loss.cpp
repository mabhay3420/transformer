#include "loss.hpp"
#include "mempool.hpp"
#include "micrograd.hpp"
MemPoolIndex mse(const std::vector<MemPoolIndex> &predicted,
                 const std::vector<MemPoolIndex> &expected,
                 MemPool<Value> *mem_pool) {
  auto mse = val(0.0f, mem_pool);
  for (int i = 0; i < predicted.size(); i++) {
    auto error = sub(predicted[i], expected[i], mem_pool);
    auto error_squared = mul(error, error, mem_pool);
    mse = add(mse, error_squared, mem_pool);
  }
  return div(mse, val(predicted.size(), mem_pool), mem_pool);
}

MemPoolIndex
cross_entropy(const std::vector<std::vector<MemPoolIndex>> &predicted,
              const std::vector<MemPoolIndex> &expected,
              MemPool<Value> *mem_pool) {
  auto total = predicted.size();
  auto error = val(0.0f, mem_pool);
  for (int i = 0; i < total; i++) {
    int correct_label = mem_pool->get(expected[i])->data;
    auto assigned_prob = predicted[i][correct_label];
    auto log_error = log(assigned_prob, mem_pool);
    error = sub(error, log_error, mem_pool);
  }
  return div(error, val(total, mem_pool), mem_pool);
}