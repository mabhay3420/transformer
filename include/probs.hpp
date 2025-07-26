#ifndef PROBS_HPP
#define PROBS_HPP

#include <vector>
#include <random>

struct MultinomialDistribution {
    std::vector<float> pdist;
    std::mt19937 gen;
    MultinomialDistribution(const std::vector<float> &pdist);
    std::vector<int> sample(int cnt);
};
#endif // PROBS_HPP