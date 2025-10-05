#include "bigramnn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <vector>

#include "dataloader.hpp"
#include "iostream"
#include "learning_rate.hpp"
#include "loss.hpp"
#include "mempool.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "neuron.hpp"
#include "optimizer.hpp"
#include "probs.hpp"
#include "tokenizer.hpp"
#include "nn.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace {

void fill_one_hot(Tensor &tensor, int row, int index) {
  if (index < 0 || index >= tensor.shape[1]) return;
  float *ptr = tensor.data();
  int stride = tensor.shape[1];
  ptr[row * stride + index] = 1.0f;
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

std::vector<float> softmax_from_logits(const float *logits, int size) {
  std::vector<float> probs(size);
  if (size == 0) return probs;
  float max_logit = logits[0];
  for (int i = 1; i < size; ++i) max_logit = std::max(max_logit, logits[i]);
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

}  // namespace

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
  auto mem_pool = new MemPool<Value>();

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
    mem_pool->deallocate_temp();
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
      mem_pool->deallocate_temp();
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
  int correct = 0;
  int total = 0;
  for (size_t i = 0; i + 1 < val_data.size(); ++i) {
    eval_input.fill(0.0f);
    fill_one_hot(eval_input, 0, val_data[i]);
    Tensor logits = model(eval_input, store);
    const float *logits_ptr = logits.data();
    int predicted = argmax_from_logits(logits_ptr, vocab_size);
    int expected = val_data[i + 1];
    if (predicted == expected) ++correct;
    ++total;
    store.clear_tape();
  }
  float accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;
  cout << "Validation accuracy: " << accuracy << endl;

  cout << "Sampled text:" << endl;
  int current = tokenizer.encode(' ');
  int total_steps = 200;
  for (int i = 0; i < total_steps; ++i) {
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
