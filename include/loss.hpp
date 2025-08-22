#pragma once

#include "mempool.hpp"
MemPoolIndex mse(const std::vector<MemPoolIndex> &predicted,
                 const std::vector<MemPoolIndex> &expected,
                 MemPool<Value> *mem_pool);

MemPoolIndex
cross_entropy(const std::vector<std::vector<MemPoolIndex>> &predicted,
              const std::vector<MemPoolIndex> &expected,
              MemPool<Value> *mem_pool);