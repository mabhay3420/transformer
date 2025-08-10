#include "bigramnn.hpp"
#include "dataloader.hpp"
#include "iostream"
#include "mempool.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "neuron.hpp"
#include "tokenizer.hpp"
#include <iomanip>
#include <vector>

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
  auto TRAIN_SIZE = 10;

  std::vector<std::vector<MemPoolIndex>> train_data_input;
  std::vector<MemPoolIndex> train_data_target;
  auto mem_pool = std::make_shared<MemPool<Value>>();
  for (int i = 0; i < TRAIN_SIZE; i++) {
    auto input_one_hot = one_hot_encode(train_data[i], vocab_size, mem_pool);
    train_data_input.push_back(input_one_hot);
    train_data_target.push_back(val(train_data[i + 1], mem_pool));
  }

  //   auto n = Neuron(vocab_size, mem_pool, false);
  auto n = Layer(vocab_size, vocab_size, mem_pool, false);

  auto r = n(train_data_input);
  for (int i = 0; i < train_data_input.size(); i++) {
    std::cout << std::setw(20) << "Input: [";
    for (auto ri : train_data_input[i]) {
      std::cout << std::setw(8) << mem_pool->get(ri)->data << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << std::setw(20) << "Target: [";
    std::cout << std::setw(8) << mem_pool->get(train_data_target[i])->data
              << " ";
    std::cout << "]" << std::endl;

    // Neuron output
    std::cout << std::setw(20) << "Neuron output: [";
    for (auto ri : r[i]) {
      std::cout << std::setw(8) << mem_pool->get(ri)->data << " ";
    }
    std::cout << "]" << std::endl;
  }
}