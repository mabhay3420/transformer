#include "bigramnn.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include "dataloader.hpp"
#include "learning_rate.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "probs.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include "train/language_utils.hpp"
#include "utils.hpp"

void BigramNNPT() {
  using std::cout;
  using std::endl;

  srand(42);
  auto text = load_text_data("data/input.txt");
  if (text.empty()) {
    cout << "No input data available" << endl;
    return;
  }

  std::set<char> unique_chars(text.begin(), text.end());
  int vocab_size = static_cast<int>(unique_chars.size());
  if (vocab_size <= 1) {
    cout << "Vocabulary too small for training" << endl;
    return;
  }
  CharTokenizer tokenizer(unique_chars);
  auto data = tokenizer.encode(text);
  if (data.size() < 2) {
    cout << "Not enough tokens" << endl;
    return;
  }

  std::vector<int> train_data;
  std::vector<int> val_data;
  split_data(0.9f, data, train_data, val_data);
  if (train_data.size() < 2 || val_data.size() < 2) {
    cout << "Insufficient train/val split" << endl;
    return;
  }

  ParameterStore store;
  store.enable_stats(true);

  constexpr int hidden_dim = 128;
  nn::Sequential model;
  model.emplace_back<nn::Linear>(vocab_size, hidden_dim, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim, hidden_dim, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim, vocab_size, store);
  auto params = model.params();

  const int base_batch = 64;
  const int batch_size = std::max(
      1, std::min<int>(base_batch, static_cast<int>(train_data.size()) - 1));
  const int epochs = 600;
  const float lr = 0.05f;

  ConstantLRScheduler scheduler(lr);
  optim::AdamW optimizer(params, scheduler, 0.9f, 0.999f, 1e-4f);

  Tensor batch_X = store.tensor({batch_size, vocab_size}, TensorInit::ZeroData);
  Tensor batch_y = store.tensor({batch_size, vocab_size}, TensorInit::ZeroData);

  std::vector<float> losses;
  store.clear_tape();
  for (int epoch = 0; epoch < epochs; ++epoch) {
    optimizer.zero_grad();
    batch_X.fill(0.0f);
    batch_y.fill(0.0f);

    for (int i = 0; i < batch_size; ++i) {
      int idx = rand() % (static_cast<int>(train_data.size()) - 1);
      int current = train_data[idx];
      int next = train_data[idx + 1];
      fill_one_hot(batch_X, i, current);
      fill_one_hot(batch_y, i, next);
    }

    Tensor logits = model(batch_X, store);
    Tensor loss = nn::bce_with_logits_loss(logits, batch_y, store);
    losses.push_back(loss.data()[0]);

    store.backward(loss);
    optimizer.step();
    store.clear_tape();

    if (epoch % 100 == 0) {
      cout << "Epoch: " << epoch << " Loss: " << losses.back() << endl;
    }
  }

  cout << "Final training loss: " << (losses.empty() ? 0.0f : losses.back())
       << endl;

  Tensor eval_input = store.tensor({1, vocab_size}, TensorInit::ZeroData);
  float accuracy = train::evaluate_sequence_accuracy(model, store, eval_input,
                                                     val_data, vocab_size);
  cout << "Validation accuracy: " << accuracy << endl;

  cout << "Sampled text:" << endl;
  int current = tokenizer.encode(' ');
  int total_steps = 200;
  for (int i = 0; i < total_steps; ++i) {
    eval_input.fill(0.0f);
    fill_one_hot(eval_input, 0, current);
    Tensor logits = model(eval_input, store);
    int next = train::sample_next_token(logits, vocab_size);
    store.clear_tape();
    current = next;
    std::cout << tokenizer.decode(current);
  }
  std::cout << std::endl;
}
