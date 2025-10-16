#include "probs.hpp"

#include <iostream>
#include <random>
#include <sstream>

MultinomialDistribution::MultinomialDistribution(
    const std::vector<float> &pdist)
    : pdist(pdist), gen(std::random_device{}()) {
  if (pdist.empty()) {
    throw std::invalid_argument("Probability distribution cannot be empty");
  }
  //   make sure probailities sum to 1
  float sum = 0.0f;
  for (auto p : pdist) {
    if (p < 0.0f || p > 1.0f) {
      throw std::invalid_argument("Probability values must be between 0 and 1");
    }
    sum += p;
  }
  if (std::abs(sum - 1.0f) > 1e-5f) {
    std::stringstream ss;
    ss << "Probability distribution must sum to 1, but has some extra"
       << (sum - 1.0f) << ". Please check your input probabilities.";
    throw std::invalid_argument(ss.str());
  }
}

std::vector<int> MultinomialDistribution::sample(int cnt) {
  std::discrete_distribution<int> dist(pdist.begin(), pdist.end());
  std::vector<int> samples(cnt);
  for (int i = 0; i < cnt; ++i) {
    samples[i] = dist(gen);
  }
  return samples;
}