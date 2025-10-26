#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using IntSequenceBatch = std::vector<std::vector<int>>;
using FloatSequenceBatch = std::vector<std::vector<float>>;
using Batch = std::pair<IntSequenceBatch, IntSequenceBatch>;

struct Sampler {
  uint32_t batch_size;
  uint32_t block_size;

  std::vector<int> train_data;
  std::vector<int> val_data;

  Sampler(size_t batch_size, size_t block_size,
          const std::vector<int>& train_data, const std::vector<int>& val_data)
      : batch_size(batch_size),
        block_size(block_size),
        train_data(train_data),
        val_data(val_data) {}

  void sample(Batch& batch, bool is_train = true);
};
