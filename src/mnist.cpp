#include "mnist.hpp"
#include "dataloader.hpp"
#include "learning_rate.hpp"
#include "mempool.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "optimizer.hpp"
#include "utils.hpp"
#include <memory>

void MnistDnn() {

  // load data
  MNIST mnist(100);
  mnist.summary();
  auto start = std::chrono::high_resolution_clock::now();
  auto mem_pool = std::make_shared<MemPool<Value>>(10000);
  auto in_size = mnist.data.train_data[0].size();
  auto out_size = 10; // 10 classes

  auto n = MLP(in_size, {128, 128, out_size}, mem_pool);
  auto params = n.params();
  auto last_param_end = mem_pool->size();
  auto TOTAL_SIZE = mnist.data.train_data.size();
  auto BATCH_SIZE = 64;
  auto TRAIN_FRACTION = 0.8;
  int VAL_SIZE = (1 - TRAIN_FRACTION) * TOTAL_SIZE;
  auto TOTAL_EPOCH = 100;
  auto TRACE_EVERY = TOTAL_EPOCH / TOTAL_EPOCH;
  TRACE_EVERY = std::max(TRACE_EVERY, 1);
  std::cout << "Total dataset size: " << TOTAL_SIZE << std::endl;
  std::cout << "Batch size: " << BATCH_SIZE << std::endl;
  std::cout << "Total epochs: " << TOTAL_EPOCH << std::endl;
  std::cout << "Total parameters: " << params.size() << std::endl;
  std::vector<float> losses;
  std::vector<std::vector<MemPoolIndex>> x_val;
  //   IMPORTANT
  mem_pool->set_persistent_boundary();
  auto LR0 = 0.1f;
  auto momentum_beta = 0.9f;
  auto LR_GAMMA = 0.1f;
  auto LR_CLIFF = 100;
  ConstantLRScheduler lr_scheduler(LR0);
  AdamOptimizer<ConstantLRScheduler> optimizer(mem_pool, params, lr_scheduler,
                                               momentum_beta);
  auto getRandomBatch = [&](int batch_size) {
    // choose a list of indices
    std::vector<int> indices(batch_size);
    auto total_size = mnist.data.train_data.size();
    for (int i = 0; i < batch_size; i++) {
      indices[i] = (rand() % (total_size - VAL_SIZE)) + VAL_SIZE;
    }
    std::vector<std::pair<std::vector<MemPoolIndex>, MemPoolIndex>> batch;
    for (int i = 0; i < batch_size; i++) {
      std::vector<MemPoolIndex> xi;
      for (auto xi_f : mnist.data.train_data[indices[i]]) {
        auto xi_v = val(xi_f, mem_pool);
        xi.push_back(xi_v);
      }
      auto yi_f = mnist.data.train_labels[indices[i]];
      auto yi_v = val(yi_f, mem_pool);
      batch.push_back({xi, yi_v});
    }
    return batch;
  };

  auto CEL = [&](std::vector<std::vector<MemPoolIndex>> predicted,
                 std::vector<MemPoolIndex> expected) {
    auto error = val(0.0f, mem_pool);
    auto total = predicted.size();
    for (auto i = 0; i < total; i++) {
      int correct_label = mem_pool->get(expected[i])->data;
      auto assigned_prob = predicted[i][correct_label];
      auto log_error = log(assigned_prob, mem_pool);
      error = sub(error, log_error, mem_pool);
    }
    auto total_examples = val(total, mem_pool);
    auto norm_error = div(error, total_examples, mem_pool);
    return norm_error;
  };

  for (int epoch = 0; epoch < TOTAL_EPOCH; epoch++) {
    mem_pool->reset();
    std::vector<float> y_val;
    auto batch = getRandomBatch(BATCH_SIZE);
    std::vector<std::vector<MemPoolIndex>> predicted;
    std::vector<MemPoolIndex> expected;
    for (auto [xi, yi] : batch) {
      auto yi_hat = n(xi);
      auto yi_hat_probs = softmax(yi_hat, mem_pool);
      predicted.push_back(yi_hat_probs);
      expected.push_back(yi);
    }
    auto loss = CEL(predicted, expected);
    losses.push_back(mem_pool->get(loss)->data);
    optimizer.zero_grad();
    backprop(loss, mem_pool);
    optimizer.step();
    if (epoch % TRACE_EVERY == 0) {
      std::cout << "Epoch: " << epoch << " Loss: " << mem_pool->get(loss)->data
                << std::endl;
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
  std::vector<std::vector<float>> x_val_float;
  std::vector<float> y_val_float;
  auto total = 0;
  auto correct = 0;
  for (int i = 0; i < VAL_SIZE; i++) {
    if (i % BATCH_SIZE == 0) {
      mem_pool->reset();
    }
    auto x_f = mnist.data.train_data[i];
    std::vector<MemPoolIndex> x;
    for (auto xi : x_f) {
      auto xi_v = val(xi, mem_pool);
      x.push_back(xi_v);
    }
    auto result_v_i = n(x);
    // what is the max index
    auto out = argmax(result_v_i, mem_pool);
    int correct_result = mnist.data.train_labels[i];
    total++;
    correct += (correct_result == out);
    y_val_float.push_back(out);
    x_val_float.push_back(x_f);
  }
  auto accuracy = static_cast<float>(correct) / total;
  std::cout << "Accuracy: " << accuracy << std::endl;
  std::cout << "Loss in the end: " << losses.back() << std::endl;
  json j = losses;
  dumpJson(j, "data/losses.json");
  dumpMemPoolEntries(params, mem_pool, "data/params.json");

  j = json{
      {"x", x_val_float},
      {"y_across_epochs", y_val_float},
  };
  dumpJson(j, "data/xor_val.json");
  mem_pool->clear();
}