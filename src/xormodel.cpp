#include "xormodel.hpp"
#include "mempool.hpp"
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
  auto mem_pool = std::make_shared<MemPool<Value>>(1000);
  auto n = MLP(2, {10, 5, 1}, mem_pool);
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
  // std::map<MemPoolIndex, float> momentum;
  std::vector<float> losses;
  std::vector<std::vector<MemPoolIndex>> x_all;
  std::vector<std::vector<MemPoolIndex>> x_val;
  std::vector<MemPoolIndex> y_all;
  for (int i = 0; i < TOTAL_SIZE; i++) {
    std::vector<MemPoolIndex> x;
    for (int j = 0; j < 2; j++) {
      auto rnd = val(get_random_float(0.0, 1.0), mem_pool);
      x.push_back(rnd);
    }
    x_all.push_back(x);
    auto first = mem_pool->get(x[0])->data > 0.5f;
    auto second = mem_pool->get(x[1])->data > 0.5f;
    // xor
    y_all.push_back(val(first ^ second, mem_pool));
  }
  auto train_fraction = 0.8;
  int train_size = train_fraction * TOTAL_SIZE;
  std::vector<std::vector<MemPoolIndex>> X;
  std::vector<MemPoolIndex> Y;
  for (int i = 0; i < train_size; i++) {
    X.push_back(x_all[i]);
    Y.push_back(y_all[i]);
  }
  for (int i = train_size; i < TOTAL_SIZE; i++) {
    x_val.push_back(x_all[i]);
  }

  // IMPORTANT
  mem_pool->set_persistent_boundary();
  std::vector<float> momentum;
  momentum.resize(mem_pool->persitent_boundary);

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
    std::vector<std::pair<std::vector<MemPoolIndex>, MemPoolIndex>> batch;
    for (int i = 0; i < batch_size; i++) {
      batch.push_back({X[indices[i]], Y[indices[i]]});
    }
    return batch;
  };

  auto MSE = [&](std::vector<MemPoolIndex> predicted,
                 std::vector<MemPoolIndex> expected) {
    auto mse = val(0.0f, mem_pool);
    for (int i = 0; i < predicted.size(); i++) {
      auto error = sub(predicted[i], expected[i], mem_pool);
      auto error_squared = mul(error, error, mem_pool);
      mse = add(mse, error_squared, mem_pool);
    }
    return div(mse, val(predicted.size(), mem_pool), mem_pool);
  };

  std::vector<std::vector<float>> y_val_epoch;
  for (int epoch = 0; epoch < TOTAL_EPOCH; epoch++) {
    mem_pool->reset();
    std::vector<float> y_val;
    auto batch = getRandomBatch(BATCH_SIZE);
    std::vector<MemPoolIndex> predicted;
    std::vector<MemPoolIndex> expected;
    for (auto [xi, yi] : batch) {
      auto yi_hat = n(xi);
      predicted.push_back(yi_hat[0]);
      // predicted.push_back(yi_hat);
      expected.push_back(yi);
    }
    auto loss = MSE(predicted, expected);
    losses.push_back(mem_pool->get(loss)->data);
    for (auto p : params) {
      mem_pool->get(p)->grad = 0.0f;
    }
    backprop(loss, mem_pool);
    for (auto p : params) {
      momentum[p] = (momentum_beta * momentum[p]) + mem_pool->get(p)->grad;
      mem_pool->get(p)->data += -LR * momentum[p];
    }
    if (epoch % TRACE_EVERY == 0) {
      auto loss_v = mem_pool->get(loss);
      std::cout << "Epoch: " << epoch << " Loss: " << loss_v->data << std::endl;
      for (auto x : x_val) {
        auto result_i = n(x)[0];
        auto result = mem_pool->get(result_i);
        if (result->data > 0.5f) {
          result_i = val(1.0f, mem_pool);
        } else {
          result_i = val(0.0f, mem_pool);
        }
        result = mem_pool->get(result_i);
        y_val.push_back(result->data);
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
  // dumpMemPoolEntries(losses, mem_pool, "data/losses.json");
  json j = losses;
  dumpJson(j, "data/losses.json");
  dumpMemPoolEntries(params, mem_pool, "data/params.json");

  std::vector<std::vector<float>> x_val_float;
  for (auto x : x_val) {
    std::vector<float> x_float;
    for (auto xi : x) {
      auto xi_v = mem_pool->get(xi);
      x_float.push_back(xi_v->data);
    }
    x_val_float.push_back(x_float);
  }
  j = json{
      {"x", x_val_float},
      {"y_across_epochs", y_val_epoch},
  };
  dumpJson(j, "data/xor_val.json");
  std::cout << "Loss in the end: " << mem_pool->get(losses.back())->data
            << std::endl;
}

void XORMLP() {}