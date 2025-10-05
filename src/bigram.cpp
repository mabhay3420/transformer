#include "bigram.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include "probs.hpp"
#include "tensor.hpp"
#include "nn.hpp"
#include "utils.hpp"
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

void BigraLm() {
  vvint table;
  vvfloat pdist;
  std::vector<MultinomialDistribution> samplers;
  auto result = load_text_data("data/input.txt");
  std::set<char> unique_chars(result.begin(), result.end());
  CharTokenizer tokenizer(unique_chars);
  auto data = tokenizer.encode(result);
  std::vector<int> train_data, val_data;
  split_data(0.9f, data, train_data, val_data);
  int block_size = 8;
  auto batch_size = 4;  // number of sequences to process in parallel
  auto sampler = Sampler(batch_size, block_size, train_data, val_data);
  auto predict_next = [&](int first) {
    auto &sampler = samplers[first];
    return sampler.sample(1)[0];
  };
  Batch trainingBatch;
  sampler.sample(trainingBatch, true);  // sample training data
  auto &[context, target] = trainingBatch;
  for (auto i = 0; i < context.size(); i++) {
    auto &x = context[i];
    auto &t = target[i];
    std::cout << i << ": #" << tokenizer.decode(x) << "# -> #"
              << tokenizer.decode(t) << "#" << std::endl;
  }
  auto vocab_size = unique_chars.size();
  table.resize(vocab_size, std::vector<int>(vocab_size, 0));
  // Build the bigram frequency table
  for (size_t i = 0; i < train_data.size() - 1; ++i) {
    int first = train_data[i];
    int second = train_data[i + 1];
    if (first < vocab_size && second < vocab_size) {
      table[first][second]++;
    }
  }
  pdist.resize(vocab_size, std::vector<float>(vocab_size, 0.0f));

  //   +1 so as to avoid zero probabilities
  for (int i = 0; i < table.size(); i++) {
    float sum = 0.0f;
    for (auto &count : table[i]) {
      sum += (count + 1);
    }
    for (int j = 0; j < table[i].size(); j++) {
      pdist[i][j] = static_cast<float>(table[i][j] + 1) / sum;
    }
  }

  //   update the distributions samplers
  samplers.clear();
  for (const auto &p : pdist) {
    samplers.emplace_back(p);
  }
  auto nll = [&](const std::vector<int> &data) {
    float loglikelihood = 0.0f;
    for (size_t i = 0; i < data.size() - 1; ++i) {
      int first = data[i];
      int second = data[i + 1];
      auto prob = pdist[first][second];
      loglikelihood += std::log(prob);
    }
    return -loglikelihood / (data.size() - 1);
  };

  float nll_train = nll(train_data);
  std::cout << "Negative log likelihood of the training data: " << nll_train
            << std::endl;
  float nll_val = nll(val_data);
  std::cout << "Negative log likelihood of the validation data: " << nll_val
            << std::endl;

  string sample1 = "Hey Wassup bro?";
  string sample2 = "Thou art a good person.";
  auto sample1_nll = nll(tokenizer.encode(sample1));
  auto sample2_nll = nll(tokenizer.encode(sample2));

  std::cout << sample1 << " : " << sample1_nll << std::endl;
  std::cout << sample2 << " : " << sample2_nll << std::endl;

  auto predictTotal = 1000;
  auto predictedToken = tokenizer.encode(' ');
  for (int i = 0; i < predictTotal; i++) {
    std::cout << tokenizer.decode(predictedToken);
    predictedToken = predict_next(predictedToken);
  }
}

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

  Tensor batch_X = store.tensor({batch_size, vocab_size}, TensorInit::ZeroData);
  Tensor batch_y = store.tensor({batch_size, vocab_size}, TensorInit::ZeroData);

  std::vector<float> losses;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    store.zero_grad();
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
    nn::sgd_step(params, lr);
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
