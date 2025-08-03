#include "xormodel.hpp"
#include "learning_rate.hpp"
#include "mempool.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "neuron.hpp"
#include "optimizer.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>

void XORLinearRegression() {

  // start time
  auto start = std::chrono::high_resolution_clock::now();
  auto mem_pool = std::make_shared<MemPool<Value>>(1000);
  auto n = MLP(2, {10, 5, 1}, mem_pool);
  auto params = n.params();
  auto last_param_end = mem_pool->size();
  std::cout << "Total params: " << params.size() << std::endl;

  auto TOTAL_SIZE = 10000;
  auto BATCH_SIZE = 64;
  auto TOTAL_EPOCH = 1000;
  auto TRACE_EVERY = TOTAL_EPOCH / 10;
  auto LR0 = 0.01f;
  // auto momentum_beta = 0.9f;
  ConstantLRScheduler lr_scheduler(LR0);
  AdamOptimizer<ConstantLRScheduler> optimizer(mem_pool, params, lr_scheduler);
  std::cout << "Total dataset size: " << TOTAL_SIZE << std::endl;
  std::cout << "Batch size: " << BATCH_SIZE << std::endl;
  std::cout << "Total epochs: " << TOTAL_EPOCH << std::endl;
  // std::cout << "Momentum beta: " << momentum_beta << std::endl;
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
  // std::vector<float> momentum;
  // momentum.resize(last_param_end);

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
  std::vector<float> val_accuracy;
  for (int epoch = 0; epoch < TOTAL_EPOCH; epoch++) {
    mem_pool->reset();
    std::vector<float> y_val;
    auto batch = getRandomBatch(BATCH_SIZE);
    std::vector<MemPoolIndex> predicted;
    std::vector<MemPoolIndex> expected;
    for (auto [xi, yi] : batch) {
      auto yi_hat = n(xi);
      predicted.push_back(yi_hat[0]);
      expected.push_back(yi);
    }
    auto loss = MSE(predicted, expected);
    losses.push_back(mem_pool->get(loss)->data);
    optimizer.zero_grad();
    backprop(loss, mem_pool);
    optimizer.step();
    if (epoch % TRACE_EVERY == 0) {
      auto total = 0;
      auto correct = 0;
      for (auto x : x_val) {
        auto result_v_i = n(x);
        auto result_i = result_v_i[0];
        auto out = mem_pool->get(result_i)->data;
        if (out > 0.5f) {
          out = 1.0f;
        } else {
          out = 0.0f;
        }
        auto first = mem_pool->get(x[0])->data;
        auto second = mem_pool->get(x[1])->data;
        auto correct_result = (first > 0.5f) ^ (second > 0.5f);
        total++;
        correct += (correct_result == out);
        y_val.push_back(out);
      }
      auto accuracy = static_cast<float>(correct) / total;
      std::cout << "Epoch: " << epoch << " Loss: " << mem_pool->get(loss)->data
                << " Accuracy: " << accuracy << std::endl;
      val_accuracy.push_back(accuracy);
      y_val_epoch.push_back(y_val);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto end_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Time taken: " << end_ms << " ms" << std::endl;
  std::cout << "Time take per epoch per batch: "
            << static_cast<float>(end_ms) / (TOTAL_EPOCH * BATCH_SIZE) << " ms"
            << std::endl;
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
  std::cout << "Val accuracy in the end: " << val_accuracy.back() << std::endl;
  std::cout << "Loss in the end: " << losses.back() << std::endl;
}

void XORMLP() {}