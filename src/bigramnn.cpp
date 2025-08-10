#include "bigramnn.hpp"
#include "dataloader.hpp"
#include "iostream"
#include "learning_rate.hpp"
#include "loss.hpp"
#include "mempool.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "neuron.hpp"
#include "optimizer.hpp"
#include "tokenizer.hpp"
#include <iomanip>
#include <vector>

struct BigramNNBatch {
  std::vector<std::vector<MemPoolIndex>> input;
  std::vector<MemPoolIndex> target;
};

void BigramNN() {
  auto result = load_text_data("data/input.txt");
  std::set<char> unique_chars(result.begin(), result.end());
  auto vocab_size = unique_chars.size();
  CharTokenizer tokenizer(unique_chars);
  auto data = tokenizer.encode(result);
  std::vector<int> train_data, val_data;
  split_data(0.9f, data, train_data, val_data);

  std::cout << "Training data size: " << train_data.size() << std::endl;
  std::cout << "Validation data size: " << val_data.size() << std::endl;
  auto mem_pool = std::make_shared<MemPool<Value>>();

  //   auto n = Neuron(vocab_size, mem_pool, false);
  auto n = Layer(vocab_size, vocab_size, mem_pool, false, false);
  auto params = n.params();
  mem_pool->set_persistent_boundary();
  auto BATCH_SIZE = 64;
  auto TOTAL_EPOCH = train_data.size() / BATCH_SIZE;
  auto LR0 = 0.001f;
  auto LR_GAMMA = 0.5f;
  auto LR_CLIFF = TOTAL_EPOCH / 5;
  //   StepLRScheduler lr_scheduler(LR0, LR_CLIFF, LR_GAMMA);
  ConstantLRScheduler lr_scheduler(LR0);
  //   AdamWOptimizer<StepLRScheduler> optimizer(mem_pool, params,
  //   lr_scheduler);
  AdamWOptimizer<ConstantLRScheduler> optimizer(mem_pool, params, lr_scheduler);

  auto getRandomBatchFn = [&]() {
    std::vector<int> indices(BATCH_SIZE);
    auto total_size = train_data.size();
    for (int i = 0; i < BATCH_SIZE; i++) {
      indices[i] = rand() % (total_size - 1);
    }
    std::vector<std::vector<MemPoolIndex>> train_data_input;
    std::vector<MemPoolIndex> train_data_target;
    for (auto i : indices) {
      auto input_one_hot = one_hot_encode(train_data[i], vocab_size, mem_pool);
      train_data_input.push_back(input_one_hot);
      train_data_target.push_back(val(train_data[i + 1], mem_pool));
    }
    return BigramNNBatch{
        train_data_input,
        train_data_target,
    };
  };

  vector<float> losses;
  std::cout << "Total epochs: " << TOTAL_EPOCH << std::endl;
  for (auto epoch = 0; epoch < TOTAL_EPOCH; epoch++) {
    optimizer.zero_grad();
    mem_pool->reset();
    std::vector<std::vector<MemPoolIndex>> predicted;
    std::vector<MemPoolIndex> expected;
    auto batch = getRandomBatchFn();
    for (int i = 0; i < batch.input.size(); i++) {
      auto r = n(batch.input[i]);
      auto probs = softmax(r, mem_pool);
      predicted.push_back(probs);
      expected.push_back(batch.target[i]);
    }
    auto loss = cross_entropy(predicted, expected, mem_pool);
    losses.push_back(mem_pool->get(loss)->data);
    backprop(loss, mem_pool);
    optimizer.step();
    std::cout << "Epoch: " << epoch << " Loss: " << mem_pool->get(loss)->data
              << std::endl;
  }

  //   Calculate validation loss
  std::vector<std::vector<MemPoolIndex>> val_data_input;
  std::vector<MemPoolIndex> val_data_target;
  for (int i = 0; i < val_data.size(); i++) {
    auto input_one_hot = one_hot_encode(val_data[i], vocab_size, mem_pool);
    val_data_input.push_back(input_one_hot);
    val_data_target.push_back(val(val_data[i + 1], mem_pool));
  }
  int total = 0;
  int correct = 0;
  int total_val_size = val_data_input.size();
  for (int i = 0; i < total_val_size; i++) {
    if (i % BATCH_SIZE == 0) {
      mem_pool->reset();
      std::cout << "Validating: " << i << "/" << val_data_input.size()
                << std::endl;
    }
    // mem_pool->reset();
    auto predicted_prob = softmax(n(val_data_input[i]), mem_pool);
    auto out = argmax(predicted_prob, mem_pool);
    int predicted_category = mem_pool->get(out)->data;
    int expected_category = mem_pool->get(val_data_target[i])->data;
    total++;
    correct += (predicted_category == expected_category);
  }
  auto accuracy = static_cast<float>(correct) / total;
  std::cout << "Val Total : " << total << std::endl;
  std::cout << "Val Correct: " << correct << std::endl;
  std::cout << "Validation accuracy: " << accuracy << std::endl;
}