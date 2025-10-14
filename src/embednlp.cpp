#include "embednlp.hpp"

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
#include "utils.hpp"

namespace {

void encode_context_row(Tensor &tensor, int row,
                        const std::vector<int> &context, int vocab_size) {
  if (tensor.shape.size() != 2) return;
  const int stride = tensor.shape[1];
  if (stride % vocab_size != 0) return;
  const int context_length = stride / vocab_size;
  float *ptr = tensor.data() + row * stride;
  for (int pos = 0; pos < context_length; ++pos) {
    const int idx = pos < static_cast<int>(context.size()) ? context[pos] : -1;
    if (idx >= 0 && idx < vocab_size) {
      ptr[pos * vocab_size + idx] = 1.0f;
    }
  }
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
    const float inv = 1.0f / std::max(1, size);
    for (int i = 0; i < size; ++i) {
      probs[i] = inv;
    }
    return probs;
  }
  for (int i = 0; i < size; ++i) {
    probs[i] /= sum;
  }
  return probs;
}

int argmax_from_logits(const float *logits, int size) {
  if (size <= 0) return 0;
  int best_idx = 0;
  float best_val = logits[0];
  for (int i = 1; i < size; ++i) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  return best_idx;
}

}  // namespace

BigramMLPData getBigramMLPData(std::vector<int> &data, int context_length,
                               int start_char_index) {
  BigramMLPData seq_data;
  for (int i = 0; i < static_cast<int>(data.size()); ++i) {
    std::vector<int> input(context_length, start_char_index);
    for (int j = 0; j < context_length; ++j) {
      const int index = i - (context_length - j);
      if (index >= 0) {
        input[j] = data[index];
      }
    }
    seq_data.input.push_back(std::move(input));
    seq_data.target.push_back(data[i]);
  }
  return seq_data;
}

void EmbedNLPPT() {
  using std::cout;
  using std::endl;

  srand(42);

  const auto text = load_text_data("data/input.txt");
  if (text.empty()) {
    cout << "No input data available" << endl;
    return;
  }

  std::set<char> unique_chars(text.begin(), text.end());
  const int vocab_size = static_cast<int>(unique_chars.size());
  if (vocab_size <= 1) {
    cout << "Vocabulary too small" << endl;
    return;
  }

  CharTokenizer tokenizer(unique_chars);
  const auto encoded = tokenizer.encode(text);
  if (encoded.empty()) {
    cout << "Failed to encode data" << endl;
    return;
  }

  std::vector<int> train_data;
  std::vector<int> val_data;
  split_data(0.9f, encoded, train_data, val_data);
  if (train_data.empty() || val_data.empty()) {
    cout << "Insufficient data after split" << endl;
    return;
  }

  constexpr int context_length = 24;
  const int start_char_index = tokenizer.encode('.');
  BigramMLPData train_seq =
      getBigramMLPData(train_data, context_length, start_char_index);
  BigramMLPData val_seq =
      getBigramMLPData(val_data, context_length, start_char_index);

  if (train_seq.input.empty()) {
    cout << "Training sequence data is empty" << endl;
    return;
  }

  const int input_dim = context_length * vocab_size;

  ParameterStore store;
  store.enable_stats(true);

  constexpr int hidden_dim = 256;
  nn::Sequential model;
  model.emplace_back<nn::Linear>(input_dim, hidden_dim, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim, hidden_dim, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim, vocab_size, store);
  auto params = model.params();

  ConstantLRScheduler scheduler(0.03f);
  optim::AdamW optimizer(params, scheduler, 0.9f, 0.999f, 1e-4f);

  const int base_batch = 128;
  const int batch_size = std::max(
      1, std::min<int>(base_batch, static_cast<int>(train_seq.input.size())));
  const int epochs = 400;

  Tensor batch_X = store.tensor({batch_size, input_dim}, TensorInit::ZeroData);
  Tensor batch_y =
      store.tensor({batch_size, vocab_size}, TensorInit::ZeroData);

  std::vector<float> losses;
  losses.reserve(epochs);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    optimizer.zero_grad();
    store.clear_tape();

    batch_X.fill(0.0f);
    batch_y.fill(0.0f);

    for (int i = 0; i < batch_size; ++i) {
      const int idx = rand() % static_cast<int>(train_seq.input.size());
      encode_context_row(batch_X, i, train_seq.input[idx], vocab_size);
      const int target = train_seq.target[idx];
      if (target >= 0 && target < vocab_size) {
        batch_y.data()[i * vocab_size + target] = 1.0f;
      }
    }

    Tensor logits = model(batch_X, store);
    Tensor loss = nn::bce_with_logits_loss(logits, batch_y, store);
    losses.push_back(loss.data()[0]);

    store.backward(loss);
    optimizer.step();

    if (epoch % 50 == 0) {
      cout << "Epoch: " << epoch << " Loss: " << losses.back() << endl;
    }
  }

  cout << "Final training loss: "
       << (losses.empty() ? 0.0f : losses.back()) << endl;

  Tensor eval_input = store.tensor({1, input_dim}, TensorInit::ZeroData);
  const size_t eval_limit = std::min<size_t>(val_seq.input.size(), 4000);
  int correct = 0;
  int total = 0;

  for (size_t i = 0; i < eval_limit; ++i) {
    eval_input.fill(0.0f);
    encode_context_row(eval_input, 0, val_seq.input[i], vocab_size);
    Tensor logits = model(eval_input, store);
    const float *logits_ptr = logits.data();
    const int predicted = argmax_from_logits(logits_ptr, vocab_size);
    const int expected = val_seq.target[i];
    if (predicted == expected) ++correct;
    ++total;
  }

  const float accuracy =
      total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 0.0f;
  cout << "Validation accuracy (" << total << " samples): " << accuracy << endl;

  cout << "Sampled text:" << endl;
  std::vector<int> context(context_length, start_char_index);
  const int total_chars = 200;
  for (int i = 0; i < total_chars; ++i) {
    eval_input.fill(0.0f);
    encode_context_row(eval_input, 0, context, vocab_size);
    Tensor logits = model(eval_input, store);
    const auto probs = softmax_from_logits(logits.data(), vocab_size);
    MultinomialDistribution dist(probs);
    const int next = dist.sample(1)[0];
    std::cout << tokenizer.decode(next);
    for (int j = 0; j < context_length - 1; ++j) {
      context[j] = context[j + 1];
    }
    context[context_length - 1] = next;
  }
  std::cout << std::endl;

  store.print_stats();
}

