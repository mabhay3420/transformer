#include "dataloader.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "neuron.hpp"
#include "utils.hpp"
#include "vis.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
void LinearRegression() {
  SwedishAutoInsurance autoInsurance;
  autoInsurance.summary();

  std::vector<std::vector<V>> X;
  std::vector<V> Y;

  // auto total_train_size = autoInsurance.data.train_data.size();
  // //   total_train_size = 2;
  // for (int i = 0; i < total_train_size; i++) {
  //   auto in = autoInsurance.data.train_data[i];
  //   auto label = autoInsurance.data.train_labels[i];
  //   X.push_back({val(in)});
  //   Y.push_back(val(label));
  // }
  // std::vector<std::vector<V>> X_Val;
  // std::vector<V> Y_Val;
  // for (int i = 0; i < autoInsurance.data.test_data.size(); i++) {
  //   X_Val.push_back({val(autoInsurance.data.test_data[i])});
  //   Y_Val.push_back(val(autoInsurance.data.test_labels[i]));
  // }
  // auto in_size = X[0].size();
  // auto out_size = 1;
  // auto n = Neuron(in_size, false);

  // auto params = n.params();
  // std::cout << "Total params: " << params.size() << std::endl;

  // std::vector<std::pair<V, V>> all_losses;
  // auto BATCH_SIZE = 16;
  // auto TOTAL_EPOCH = 100;
  // auto LR0 = 0.0001f;
  // auto LR = LR0;
  // auto momentum_beta = 0.2f;
  // std::map<V, float> momentum;
  // auto getRandomBatch = [&](int batch_size) {
  //   // choose a list of indices
  //   std::vector<int> indices(batch_size);
  //   for (int i = 0; i < batch_size; i++) {
  //     indices[i] = rand() % X.size();
  //   }
  //   std::vector<std::pair<std::vector<V>, V>> batch;
  //   for (int i = 0; i < batch_size; i++) {
  //     batch.push_back({X[indices[i]], Y[indices[i]]});
  //   }
  //   return batch;
  // };

  // auto MSE = [](const std::vector<V> &a, const std::vector<V> &b) {
  //   auto squared_error = val(0.0f);
  //   for (int i = 0; i < a.size(); i++) {
  //     auto diff = a[i] - b[i];
  //     squared_error = squared_error + (diff * diff);
  //   }
  //   return squared_error / a.size();
  // };

  // for (int epoch = 0; epoch < TOTAL_EPOCH; epoch++) {
  //   auto batch = getRandomBatch(BATCH_SIZE);
  //   std::vector<V> predicted;
  //   std::vector<V> expected;
  //   for (auto [xi, yi] : batch) {
  //     auto yi_hat = n(xi);
  //     predicted.push_back(yi_hat);
  //     expected.push_back(yi);
  //   }
  //   auto total_loss = MSE(predicted, expected);
  //   //   backprop
  //   for (auto p : params) {
  //     p->grad = 0.0f;
  //   }
  //   backprop(total_loss);
  //   //   update
  //   for (auto p : params) {
  //     momentum[p] = (momentum_beta * momentum[p]) + p->grad;
  //     p->data += -LR * momentum[p];
  //   }
  //   //   print predictions
  //   if (epoch == TOTAL_EPOCH - 1) {
  //     auto g = trace(total_loss);
  //     string filename = "data/vis2.dot";
  //     to_dot(g, filename);
  //     print_v(expected);
  //     print_v(predicted);
  //   }
  //   std::vector<V> predicted_Val;
  //   std::vector<V> expected_Val;
  //   for (int i = 0; i < X_Val.size(); i++) {
  //     auto predicted = n(X_Val[i]);
  //     predicted_Val.push_back(predicted);
  //     expected_Val.push_back(Y_Val[i]);
  //   }
  //   auto val_loss = MSE(predicted_Val, expected_Val);
  //   all_losses.push_back({total_loss, val_loss});
  // }

  // cout << "Final RMSE on validation data: "
  //      << std::sqrt(all_losses.back().second->data) << endl;

  // json j = all_losses;
  // auto filename = "data/losses.json";
  // std::ofstream file(filename);
  // if (!file.is_open()) {
  //   std::cout << "Could not open file for writing: " << filename << std::endl;
  //   throw std::runtime_error("FILE_NOT_FOUND");
  // }
  // file << j.dump(4) << std::endl;
  // file.close();

  // json j2 = json{{"X_Train", X},
  //                {"Y_Train", Y},
  //                {"X_Val", X_Val},
  //                {"Y_Val", Y_Val},
  //                {"params", params}};
  // auto filename2 = "data/params.json";
  // std::ofstream file2(filename2);
  // if (!file2.is_open()) {
  //   std::cout << "Could not open file for writing: " << filename2 << std::endl;
  //   throw std::runtime_error("FILE_NOT_FOUND");
  // }
  // file2 << j2.dump(4) << std::endl;
  // file2.close();
}