#pragma once

#include <vector>

void EmbedNLP();
struct BigramMLPData {
  std::vector<std::vector<int>> input;
  std::vector<int> target;
};

BigramMLPData getBigramMLPData(std::vector<int> &data, int context_length = 0,
                               int start_char_index = 0);