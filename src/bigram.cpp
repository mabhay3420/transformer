#include "bigram.hpp"
#include "probs.hpp"
#include <numeric>

BigramLM::BigramLM(int vocab_size, const std::vector<int> &train_data)
    : vocab_size(vocab_size) {
  table.resize(vocab_size, std::vector<int>(vocab_size, 0));
  // Build the bigram frequency table
  for (size_t i = 0; i < train_data.size() - 1; ++i) {
    int first = train_data[i];
    int second = train_data[i + 1];
    if (first < vocab_size && second < vocab_size) {
      table[first][second]++;
    }
  }
  pdist.resize(vocab_size, std::vector<float>(vocab_size, 0.0f));

  //   +1 so as to avoid zero probabilities
  for (int i = 0; i < table.size(); i++) {
    float sum = 0.0f;
    for (auto &count : table[i]) {
      sum += (count + 1);
    }
    for (int j = 0; j < table[i].size(); j++) {
      pdist[i][j] = static_cast<float>(table[i][j] + 1) / sum;
    }
  }

  //   update the distributions samplers
  samplers.clear();
  for (const auto &p : pdist) {
    samplers.emplace_back(p);
  }
}

int BigramLM::predict_next(int first) {
  // probability distribution for the next word given the first word
  //   auto p = pdist[first];
  auto &sampler = samplers[first];
  return sampler.sample(1)[0];
}

float BigramLM::nll(const std::vector<int> &data) {
  float loglikelihood = 0.0f;
  for (size_t i = 0; i < data.size() - 1; ++i) {
    int first = data[i];
    int second = data[i + 1];
    auto prob = pdist[first][second];
    loglikelihood += std::log(prob);
  }
  return -loglikelihood / (data.size() - 1);
}