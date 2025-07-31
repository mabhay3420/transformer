#include "xormodel.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "neuron.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>

void XORLinearRegression() {

  // auto n = Neuron(2, false);
  // auto n = Neuron(2, true);
  // start time
  auto start = std::chrono::high_resolution_clock::now();
  auto n = MLP(2, {10, 5, 1});
  auto params = n.params();
  std::cout << "Total params: " << params.size() << std::endl;

  auto get_random_float = [&](float min, float max) {
    return (float)rand() / RAND_MAX * (max - min) + min;
  };
  auto TOTAL_SIZE = 10000;
  // auto TOTAL_SIZE = 50;
  auto BATCH_SIZE = 16;
  auto TOTAL_EPOCH = 10000;
  // auto TOTAL_EPOCH = 10;
  auto TRACE_EVERY = TOTAL_EPOCH / 10;
  auto LR0 = 0.01f;
  auto LR = LR0;
  auto momentum_beta = 0.0f;
  std::map<V, float> momentum;
  std::vector<V> losses;
  std::vector<std::vector<V>> x_all;
  std::vector<std::vector<V>> x_val;
  std::vector<V> y_all;
  for (int i = 0; i < TOTAL_SIZE; i++) {
    std::vector<V> x;
    for (int j = 0; j < 2; j++) {
      auto rnd = val(get_random_float(0.0, 1.0));
      x.push_back(rnd);
    }
    x_all.push_back(x);
    auto first = x[0]->data > 0.5f;
    auto second = x[1]->data > 0.5f;
    // xor
    y_all.push_back(val(first ^ second));
  }
  auto train_fraction = 0.8;
  int train_size = train_fraction * TOTAL_SIZE;
  std::vector<std::vector<V>> X;
  std::vector<V> Y;
  for (int i = 0; i < train_size; i++) {
    X.push_back(x_all[i]);
    Y.push_back(y_all[i]);
  }
  for (int i = train_size; i < TOTAL_SIZE; i++) {
    x_val.push_back(x_all[i]);
  }
  // {
  //     {val(0.0), val(1.0)},
  //     {val(1.0), val(0.0)},
  //     {val(1.0), val(1.0)},
  //     {val(0.0), val(0.0)},
  // };

  // = {val(1.0), val(1.0), val(0.0), val(0.0)};

  auto getRandomBatch = [&](int batch_size) {
    // choose a list of indices
    std::vector<int> indices(batch_size);
    for (int i = 0; i < batch_size; i++) {
      indices[i] = rand() % X.size();
    }
    std::vector<std::pair<std::vector<V>, V>> batch;
    for (int i = 0; i < batch_size; i++) {
      batch.push_back({X[indices[i]], Y[indices[i]]});
    }
    return batch;
  };

  auto MSE = [&](std::vector<V> predicted, std::vector<V> expected) {
    auto mse = val(0.0f);
    for (int i = 0; i < predicted.size(); i++) {
      mse = mse + (predicted[i] - expected[i]) * (predicted[i] - expected[i]);
    }
    return mse / predicted.size();
  };

  std::vector<std::vector<V>> y_val_epoch;
  for (int epoch = 0; epoch < TOTAL_EPOCH; epoch++) {
    std::vector<V> y_val;
    auto batch = getRandomBatch(BATCH_SIZE);
    std::vector<V> predicted;
    std::vector<V> expected;
    for (auto [xi, yi] : batch) {
      auto yi_hat = n(xi);
      predicted.push_back(yi_hat[0]);
      // predicted.push_back(yi_hat);
      expected.push_back(yi);
    }
    auto loss = MSE(predicted, expected);
    losses.push_back(val(loss->data));
    for (auto p : params) {
      p->grad = 0.0f;
    }
    backprop(loss);
    for (auto p : params) {
      momentum[p] = (momentum_beta * momentum[p]) + p->grad;
      p->data += -LR * momentum[p];
    }
    if (epoch % TRACE_EVERY == 0) {
      std::cout << "Epoch: " << epoch << " Loss: " << loss->data << std::endl;
      for (auto x : x_val) {
        auto result = n(x)[0];
        if (result->data > 0.5f) {
          result = val(1.0f);
        } else {
          result = val(0.0f);
        }
        y_val.push_back(result);
      }
      y_val_epoch.push_back(y_val);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
  dumpValues(losses, "data/losses.json");
  dumpValues(params, "data/params.json");
  json j = json{
      {"x", x_val},
      {"y_across_epochs", y_val_epoch},
  };
  dumpJson(j, "data/xor_val.json");
  std::cout << "Loss in the end: " << losses.back() << std::endl;
}

void XORMLP() {}