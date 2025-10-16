#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename T>
void split_data(const float ratio, const std::vector<T>& data,
                std::vector<T>& train_data, std::vector<T>& val_data) {
  assert(ratio > 0.0f && ratio < 1.0f && "Ratio must be between 0 and 1");
  auto split_index = static_cast<size_t>(data.size() * ratio);
  train_data.assign(data.begin(), data.begin() + split_index);
  val_data.assign(data.begin() + split_index, data.end());
  if (train_data.empty() || val_data.empty()) {
    throw std::runtime_error("Split resulted in empty dataset");
  }
}
