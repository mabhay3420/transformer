#include "data/sampler.hpp"

#include <stdlib.h>

#include <cstdlib>
#include <vector>

void Sampler::sample(Batch& batch, bool is_train) {
  auto get_random_index = [](int size) {
    unsigned seed = 1234;
    return rand_r(&seed) % size;
  };
  auto data_vec = is_train ? train_data : val_data;
  for (uint32_t i = 0; i < batch_size; ++i) {
    unsigned seed = batch_size + i;
    auto rnd_idx =
        rand_r(&seed) % (static_cast<int>(data_vec.size()) - block_size);
    auto train_seq = std::vector<int>(data_vec.begin() + rnd_idx,
                                      data_vec.begin() + rnd_idx + block_size);
    auto target = std::vector<int>(data_vec.begin() + rnd_idx + 1,
                                   data_vec.begin() + rnd_idx + block_size + 1);
    batch.first.push_back(train_seq);
    batch.second.push_back(target);
  }
}
