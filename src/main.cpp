#include "bigram.hpp"
#include "dataloader.hpp"
#include "linear_regression.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "mnist.hpp"
#include "neuron.hpp"
#include "probs.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include "vis.hpp"
#include "xormodel.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>

#define LABEL(I) I->label = #I

float f(float x) { return 3 * x * x - 4 * x + 5; }
float df(float x) {
  float h = 0.00001f;
  return (f(x + h) - f(x - h)) / (2 * h);
}
float dfreal(float x) {
  //   return 6 * x - 4;
  return 6 * x - 4;
}

int main() {
  //   auto result = load_text_data("data/input.txt");
  //   std::set<char> unique_chars(result.begin(), result.end());
  //   CharTokenizer tokenizer(unique_chars);
  //   std::vector<int> data;
  //   tokenizer.encode(result, data);
  //   std::vector<int> train_data, val_data;
  //   split_data(0.9f, data, train_data, val_data);
  //   // context length
  //   int block_size = 8;
  //   auto batch_size = 4; // number of sequences to process in parallel
  //   auto sampler = Sampler(batch_size, block_size, train_data, val_data);
  //   Batch trainingBatch;
  //   sampler.sample(trainingBatch, true); // sample training data
  //   auto vocab_size = unique_chars.size();
  //   BigramLM bigramLM(vocab_size, train_data);
  //   float nll_train = bigramLM.nll(train_data);
  //   std::cout << "Negative log likelihood of the training data: " <<
  //   nll_train
  //             << std::endl;

  //   //   Maximize the likelihood
  //   // == Maximize the log likelihood
  //   // == Minimize the negative log likelihood
  //   // == Minimize the loss wrt the probability distribution

  //   //   check the loss on validation data
  //   float nll_val = bigramLM.nll(val_data);
  //   auto total_samples = val_data.size() - 1;
  //   std::cout << "Negative log likelihood of the validation data: " <<
  //   nll_val
  //             << std::endl;

  //   //   A simple test to whether a sentence is shakespearian or not
  //   //   Lower NLL means the sentence is more likely to be shakespearian
  //   string sample1 = "Hey Wassup bro?";
  //   string sample2 = "Thou art a good person.";
  //   std::vector<int> encoded1, encoded2;
  //   tokenizer.encode(sample1, encoded1);
  //   tokenizer.encode(sample2, encoded2);

  //   auto sample1_nll = bigramLM.nll(encoded1);
  //   auto sample2_nll = bigramLM.nll(encoded2);

  //   std::cout << sample1 << " : " << sample1_nll << std::endl;
  //   std::cout << sample2 << " : " << sample2_nll << std::endl;

  // LinearRegression();
  // XORLinearRegression();
  MnistDnn();
  return 0;
}