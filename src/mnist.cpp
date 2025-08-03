#include "mnist.hpp"
#include "dataloader.hpp"
#include "learning_rate.hpp"
#include "loss.hpp"
#include "mempool.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "optimizer.hpp"
#include "utils.hpp"
#include <memory>

void MnistDnn() {

  auto LINES_TO_READ = 60;
  MNIST mnist(LINES_TO_READ);
  mnist.summary();
  auto start = std::chrono::high_resolution_clock::now();
  auto mem_pool = std::make_shared<MemPool<Value>>();
  auto in_size = mnist.data.train_data[0].size();
  auto out_size = 10; // 10 classes

  auto n = MLP(in_size, {128, 128, out_size}, mem_pool);
  mem_pool->set_persistent_boundary();

  auto params = n.params();
  auto TOTAL_SIZE = mnist.data.train_data.size();
  auto BATCH_SIZE = 64;
  auto TRAIN_FRACTION = 0.8;
  int VAL_SIZE = (1 - TRAIN_FRACTION) * TOTAL_SIZE;
  auto TOTAL_EPOCH = 10000;
  auto TRACE_EVERY = TOTAL_EPOCH / TOTAL_EPOCH;
  TRACE_EVERY = std::max(TRACE_EVERY, 1);

  std::cout << "Total dataset size: " << TOTAL_SIZE << std::endl;
  std::cout << "Batch size: " << BATCH_SIZE << std::endl;
  std::cout << "Total epochs: " << TOTAL_EPOCH << std::endl;
  std::cout << "Total parameters: " << params.size() << std::endl;

  std::vector<float> losses;
  //   IMPORTANT
  auto LR0 = 0.01f;
  auto LR_GAMMA = 0.1f;
  auto LR_CLIFF = 100;
  ConstantLRScheduler lr_scheduler(LR0);
  AdamOptimizer<ConstantLRScheduler> optimizer(mem_pool, params, lr_scheduler);

  auto inputTransform = [&](const std::vector<float> &x) {
    return val(x, mem_pool);
  };
  auto labelTransform = [&](const float x) { return val(x, mem_pool); };
  auto getRandomBatchFn =
      std::bind(getRandomBatch<std::vector<float>, float,
                               std::vector<MemPoolIndex>, MemPoolIndex>,
                mnist.data.train_data, mnist.data.train_labels, inputTransform,
                labelTransform, BATCH_SIZE, VAL_SIZE, -1);

  // TRAINING LOOP
  for (int epoch = 0; epoch < TOTAL_EPOCH; epoch++) {
    mem_pool->reset();
    auto batch = getRandomBatchFn();
    std::vector<std::vector<MemPoolIndex>> predicted;
    std::vector<MemPoolIndex> expected;
    for (auto [xi, yi] : batch) {
      auto yi_hat = n(xi);
      auto yi_hat_probs = softmax(yi_hat, mem_pool);
      predicted.push_back(yi_hat_probs);
      expected.push_back(yi);
    }
    auto loss = cross_entropy(predicted, expected, mem_pool);
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
    auto x = inputTransform(x_f);
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