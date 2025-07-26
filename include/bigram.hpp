#ifndef BIGRAM_HPP
#define BIGRAM_HPP
#include "dataloader.hpp"
#include "probs.hpp"

struct BigramLM {

  int vocab_size;
  vvint table;
  vvfloat pdist;
  std::vector<MultinomialDistribution> samplers;
  BigramLM(int vocab_size, const std::vector<int> &train_data);
  int predict_next(int first);
  float nll(const std::vector<int> &data);
};

#endif // BIGRAM_HPP