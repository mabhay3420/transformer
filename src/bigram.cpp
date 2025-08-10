#include "bigram.hpp"
#include "probs.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <set>

void BigraLm() {
  vvint table;
  vvfloat pdist;
  std::vector<MultinomialDistribution> samplers;
  auto result = load_text_data("data/input.txt");
  std::set<char> unique_chars(result.begin(), result.end());
  CharTokenizer tokenizer(unique_chars);
  std::vector<int> data;
  tokenizer.encode(result, data);
  std::vector<int> train_data, val_data;
  split_data(0.9f, data, train_data, val_data);
  int block_size = 8;
  auto batch_size = 4; // number of sequences to process in parallel
  auto sampler = Sampler(batch_size, block_size, train_data, val_data);
  // auto inputTransform = [&](const std::vector<float> &x) {
  //   return val(x, mem_pool);
  // };
  // auto labelTransform = [&](const float x) { return val(x, mem_pool); };
  // auto getRandomBatchFn =
  //     std::bind(getRandomBatch<std::vector<float>, float,
  //                              std::vector<MemPoolIndex>, MemPoolIndex>,
  //               mnist.data.train_data, mnist.data.train_labels,
  //               inputTransform, labelTransform, BATCH_SIZE, VAL_SIZE, -1);
  auto predict_next = [&](int first) {
    // probability distribution for the next word given the first word
    //   auto p = pdist[first];
    auto &sampler = samplers[first];
    return sampler.sample(1)[0];
  };
  Batch trainingBatch;
  sampler.sample(trainingBatch, true); // sample training data
  auto vocab_size = unique_chars.size();
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
  auto nll = [&](const std::vector<int> &data) {
    float loglikelihood = 0.0f;
    for (size_t i = 0; i < data.size() - 1; ++i) {
      int first = data[i];
      int second = data[i + 1];
      auto prob = pdist[first][second];
      loglikelihood += std::log(prob);
    }
    return -loglikelihood / (data.size() - 1);
  };

  float nll_train = nll(train_data);
  std::cout << "Negative log likelihood of the training data: " << nll_train
            << std::endl;
  float nll_val = nll(val_data);
  // auto total_samples = val_data.size() - 1;
  std::cout << "Negative log likelihood of the validation data: " << nll_val
            << std::endl;

  string sample1 = "Hey Wassup bro?";
  string sample2 = "Thou art a good person.";
  std::vector<int> encoded1, encoded2;
  tokenizer.encode(sample1, encoded1);
  tokenizer.encode(sample2, encoded2);

  auto sample1_nll = nll(encoded1);
  auto sample2_nll = nll(encoded2);

  std::cout << sample1 << " : " << sample1_nll << std::endl;
  std::cout << sample2 << " : " << sample2_nll << std::endl;
}