#include "bigram.hpp"
#include "dataloader.hpp"
#include "linear_regression.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
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

  // std::vector<std::vector<V>> x = {
  //     {val(2.0f), val(3.0f), val(-1.0f)},
  //     {val(3.0f), val(-1.0f), val(-0.5f)},
  //     {val(-0.5f), val(1.0f), val(1.0f)},
  //     {val(-1.0f), val(1.0f), val(1.0f)},
  // };
  // std::vector<V> y = {val(1.0f), val(-1.0f), val(-1.0f), val(1.0f)};
  // MNIST mnist;
  // mnist.summary();
  // std::vector<std::vector<V>> x;
  // std::vector<std::vector<V>> y;
  // auto in_size = mnist.data.train_data[0].size();
  // auto out_size = 10; // 10 classes
  // int total_train_inputs = 100;
  // // in_size = 1;
  // // out_size = 1;"data/vis2.dot")
  // total_train_inputs = 1;
  // for (int i = 0; i < mnist.data.train_data.size(); i++) {
  //   // TEMP
  //   if (i >= total_train_inputs)
  //     break;
  //   auto in = mnist.data.train_data[i];
  //   auto label = mnist.data.train_labels[i];
  //   std::vector<V> xi;
  //   for (int j = 0; j < in_size; j++) {
  //     xi.push_back(val(in[j]));
  //   }
  //   x.push_back(xi);
  //   y.push_back(one_hot_encode(int(label), out_size));
  // }
  // auto n = MLP(in_size, {out_size});
  // // auto n = MLP(3, {4, 1});
  // auto params = n.params();
  // for (auto p : params) {
  //   p->is_param = true;
  // }
  // // print_v(params);
  // std::cout << "Total params: " << params.size() << std::endl;
  // std::vector<V> topo;
  // std::vector<V> Losses;
  // int train_loop_size = 10;
  // int BATCH_SIZE = 16;
  // float lr0 = 0.01f;
  // float clip = 1.0f;
  // float decay = 0.5f;
  // int decay_step = 10;
  // for (int t = 0; t < train_loop_size; t++) {
  //   // float lr = lr0 * std::pow(decay, t / decay_step);
  //   for (int j = 0; j < x.size(); j += BATCH_SIZE) {
  //     std::vector<V> z;
  //     std::vector<V> l;
  //     auto batch_size = std::min<int>(BATCH_SIZE, x.size() - j);
  //     // going through training data
  //     for (int i = j; i < j + batch_size; i++) {
  //       auto xi = x[i];
  //       // for (int p = 0; p < xi.size(); p++) {
  //       //   xi[p]->label = std::to_string(t) + "_x_" + std::to_string(i) +
  //       "_"
  //       //   +
  //       //                  std::to_string(p);
  //       // }
  //       auto yi = y[i];
  //       // yi->label = std::to_string(t) + "_y_" + std::to_string(i);

  //       if (xi.size() != in_size || yi.size() != out_size) {
  //         std::cout << "xi.size() = " << xi.size() << std::endl;
  //         std::cout << "yi.size() = " << yi.size() << std::endl;
  //         std::cout << "in_size = " << in_size << std::endl;
  //         std::cout << "out_size = " << out_size << std::endl;
  //         throw std::runtime_error("Invalid input size");
  //       }
  //       // loss calculations
  //       auto softmax_result = softmax(n(xi));
  //       assert(softmax_result.size() == out_size);
  //       auto logit = val(0.0);
  //       int predicted_category = 0;
  //       auto max = val(0.0);
  //       for (int k = 0; k < out_size; k++) {
  //         if (softmax_result[k] > max) {
  //           max = softmax_result[k];
  //           predicted_category = k;
  //         }
  //         logit = logit - (yi[k] * log(softmax_result[k]));
  //       }
  //       auto cross_entropy_loss = logit / out_size;
  //       // auto predicted_category = n(xi)[0];
  //       // predicted_category->label = "predicted_category";
  //       // auto cross_entropy_loss =
  //       //     (predicted_category - yi) * (predicted_category - yi);
  //       cross_entropy_loss->label = "cross_entropy_loss";
  //       // z.push_back(predicted_category);
  //       z.push_back(val(predicted_category));
  //       l.push_back(cross_entropy_loss);
  //     }
  //     // if (t == train_loop_size - 1) {
  //     //   std::cout << "Targets: " << y << std::endl;
  //     //   std::cout << "Predictions: " << z << std::endl;
  //     // }

  //     // std::cout << "Losses: " << l << std::endl;
  //     auto L = val(0.0f);
  //     for (int i = 0; i < l.size(); i++) {
  //       L = L + l[i];
  //     }
  //     L = L / batch_size;
  //     L->label = "Total Loss";
  //     std::cout << t << "[" << j / BATCH_SIZE << "]"
  //               << ": " << L << std::endl;
  //     // set the gradients to zero
  //     for (auto param : params) {
  //       param->grad = 0.0f;
  //     }
  //     // calculate gradients
  //     backprop(L);
  //     for (auto param : params) {
  //       if (param->grad > 0.0f || param->grad < 0.0f) {
  //         std::cout << param->data << ": " << param->grad << std::endl;
  //       }
  //     }
  //     Losses.push_back(L);
  //     if (j == 0 && t == 0) {
  //       auto g = trace(L);
  //       string filename = "data/vis2.dot";
  //       to_dot(g, filename);
  //     }

  //     // update params according to there gradients
  //     for (auto param : params) {
  //       // if (param->grad > clip)
  //       //   param->grad = clip;
  //       // if (param->grad < -clip)
  //       //   param->grad = -clip;
  //       // param->data += -lr * param->grad;
  //       param->data += -lr0 * param->grad;
  //     }
  //     if (t == train_loop_size - 1) {
  //       print_v(z);
  //       print_v(y);
  //     }
  //   }
  // }
  // // print_vv(x);
  // // Losses
  // json j = Losses;
  // auto filename = "data/losses.json";
  // std::ofstream file(filename);
  // if (!file.is_open()) {
  //   std::cout << "Could not open file for writing: " << filename <<
  //   std::endl; throw std::runtime_error("FILE_NOT_FOUND");
  // }
  // file << j.dump(4) << std::endl;
  // file.close();

  // LinearRegression();
  XORLinearRegression();
  return 0;
}