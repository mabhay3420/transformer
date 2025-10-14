#include "bigram.hpp"

#include <algorithm>
#include <cmath>
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

namespace {

void fill_one_hot(Tensor &tensor, int row, int index) {
  if (index < 0 || index >= tensor.shape[1]) return;
  float *ptr = tensor.data();
  int stride = tensor.shape[1];
  ptr[row * stride + index] = 1.0f;
}

std::vector<float> softmax_from_logits(const float *logits, int size) {
  std::vector<float> probs(size);
  if (size == 0) return probs;
  float max_logit = logits[0];
  for (int i = 1; i < size; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    float val = std::exp(logits[i] - max_logit);
    probs[i] = val;
    sum += val;
  }
  if (sum <= 0.0f) {
    float inv = 1.0f / std::max(1, size);
    for (int i = 0; i < size; ++i) probs[i] = inv;
    return probs;
  }
  for (int i = 0; i < size; ++i) probs[i] /= sum;
  return probs;
}

float evaluate_nll(nn::Sequential &model, ParameterStore &store,
                   Tensor &scratch_input, const std::vector<int> &sequence,
                   int vocab_size) {
  if (sequence.size() < 2) return 0.0f;
  float total = 0.0f;
  scratch_input.fill(0.0f);
  for (size_t i = 0; i + 1 < sequence.size(); ++i) {
    scratch_input.fill(0.0f);
    fill_one_hot(scratch_input, 0, sequence[i]);
    Tensor logits = model(scratch_input, store);
    const float *logits_ptr = logits.data();
    auto probs = softmax_from_logits(logits_ptr, vocab_size);
    float prob = std::max(probs[sequence[i + 1]], 1e-8f);
    total += -std::log(prob);
    store.clear_tape();
  }
  return total / static_cast<float>(sequence.size() - 1);
}

}  // namespace

void BigraLmPT() {
  using std::cout;
  using std::endl;

  srand(42);

  auto text = load_text_data("data/input.txt");
  if (text.empty()) {
    cout << "No input data available" << endl;
    return;
  }

  std::set<char> unique_chars(text.begin(), text.end());
  CharTokenizer tokenizer(unique_chars);
  auto encoded = tokenizer.encode(text);
  if (encoded.size() < 2) {
    cout << "Not enough data to train bigram model" << endl;
    return;
  }

  std::vector<int> train_data;
  std::vector<int> val_data;
  split_data(0.9f, encoded, train_data, val_data);
  if (train_data.size() < 2 || val_data.size() < 2) {
    cout << "Insufficient train/val split" << endl;
    return;
  }

  int vocab_size = static_cast<int>(unique_chars.size());
  ParameterStore store;
  store.enable_stats(true);

  constexpr int hidden_dim = 64;
  nn::Sequential model;
  model.emplace_back<nn::Linear>(vocab_size, hidden_dim, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim, vocab_size, store);
  auto params = model.params();

  const int base_batch = 128;
  const int batch_size = std::max(1, std::min<int>(base_batch,
                                                   static_cast<int>(train_data.size()) - 1));
  const int epochs = 500;
  const float lr = 0.1f;

  ConstantLRScheduler scheduler(lr);
  optim::SGD optimizer(params, scheduler);

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
  float train_nll = evaluate_nll(model, store, eval_input, train_data, vocab_size);
  float val_nll = evaluate_nll(model, store, eval_input, val_data, vocab_size);
  cout << "Training NLL: " << train_nll << endl;
  cout << "Validation NLL: " << val_nll << endl;

  cout << "Sampled text:" << endl;
  int current = tokenizer.encode(' ');
  int total = 200;
  for (int i = 0; i < total; ++i) {
    eval_input.fill(0.0f);
    fill_one_hot(eval_input, 0, current);
    Tensor logits = model(eval_input, store);
    const float *logits_ptr = logits.data();
    auto probs = softmax_from_logits(logits_ptr, vocab_size);
    store.clear_tape();
    MultinomialDistribution dist(probs);
    current = dist.sample(1)[0];
    std::cout << tokenizer.decode(current);
  }
  std::cout << std::endl;
}
