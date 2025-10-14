#include "mnist.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>

#include "dataloader.hpp"
#include "learning_rate.hpp"
#include "loss.hpp"
#include "mempool.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace {

void fill_one_hot(Tensor &tensor, int row, int index) {
  if (tensor.shape.size() != 2) return;
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

}  // namespace

void MnistDnn() {
  auto LINES_TO_READ = 60000;
  MNIST mnist(LINES_TO_READ);
  mnist.summary();
  auto start = std::chrono::high_resolution_clock::now();
  auto mem_pool = new MemPool<Value>();
  auto in_size = mnist.data.train_data[0].size();
  auto out_size = 10;  // 10 classes

  auto n = MLP(in_size, {10, out_size}, mem_pool, false);
  mem_pool->set_persistent_boundary();

  auto params = n.params();
  auto TOTAL_SIZE = mnist.data.train_data.size();
  auto BATCH_SIZE = 64;
  auto TRAIN_FRACTION = 0.8;
  // int VAL_SIZE =
  //     std::min<int>((1 - TRAIN_FRACTION) * TOTAL_SIZE, BATCH_SIZE * 4);
  int VAL_SIZE = (1 - TRAIN_FRACTION) * TOTAL_SIZE;
  // auto TOTAL_EPOCH = TOTAL_SIZE / BATCH_SIZE;
  auto TOTAL_EPOCH = 1000;
  auto TRACE_EVERY = TOTAL_EPOCH / TOTAL_EPOCH;
  TRACE_EVERY = std::max<int>(TRACE_EVERY, 1);

  std::cout << "Total dataset size: " << TOTAL_SIZE << std::endl;
  std::cout << "Batch size: " << BATCH_SIZE << std::endl;
  std::cout << "Total epochs: " << TOTAL_EPOCH << std::endl;
  std::cout << "Total parameters: " << params.size() << std::endl;

  std::vector<float> losses;
  //   IMPORTANT
  auto LR0 = 0.01f;
  auto LR_GAMMA = 0.5f;
  auto LR_CLIFF = TOTAL_EPOCH / 5;
  StepLRScheduler lr_scheduler(LR0, LR_CLIFF, LR_GAMMA);
  // ConstantLRScheduler lr_scheduler(LR0);
  AdamWOptimizer<StepLRScheduler> optimizer(mem_pool, params, lr_scheduler);

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
    optimizer.zero_grad();
    mem_pool->deallocate_temp();
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
  std::cout << "Time take per epoch: "
            << static_cast<float>(end_ms) / TOTAL_EPOCH << " ms" << std::endl;
  std::vector<std::vector<float>> x_val_float;
  std::vector<float> y_val_float;
  auto total = 0;
  auto correct = 0;
  for (int i = 0; i < VAL_SIZE; i++) {
    if (i % BATCH_SIZE == 0) {
      std::cout << "Validating: " << i << "/" << VAL_SIZE << std::endl;
      mem_pool->deallocate_temp();
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
  mem_pool->deallocate_all();
}

void MnistDnnPT() {
  using std::cout;
  using std::endl;

  srand(42);

  const int MAX_TRAIN_SAMPLES = 60000;
  const int MAX_TEST_SAMPLES = 5000;
  const int epochs = 15;
  // const int MAX_TRAIN_SAMPLES = 6000;
  // const int MAX_TEST_SAMPLES = 500;
  // const int epochs = 5;
  const int hidden_dim1 = 256;
  const int hidden_dim2 = 128;
  const int num_classes = 10;
  const int batch_size = 128;
  const int eval_batch = 128;
  const float lr = 0.01f;

  MNIST mnist(MAX_TRAIN_SAMPLES);
  mnist.summary();

  int input_dim = static_cast<int>(mnist.data.train_data[0].size());

  int total_samples =
      std::min<int>(mnist.data.train_data.size(), MAX_TRAIN_SAMPLES);
  const float train_fraction = 0.85f;
  int train_count =
      std::max<int>(1, static_cast<int>(total_samples * train_fraction));
  train_count = std::min(train_count, total_samples);
  int val_count = total_samples - train_count;
  const int test_total =
      std::min<int>(mnist.data.test_data.size(), MAX_TEST_SAMPLES);
  const int steps_per_epoch = std::max(1, train_count / batch_size);

  ParameterStore store;
  store.enable_stats(true);

  const size_t param_elements =
      static_cast<size_t>(input_dim) * hidden_dim1 + hidden_dim1 +
      static_cast<size_t>(hidden_dim1) * hidden_dim2 + hidden_dim2 +
      static_cast<size_t>(hidden_dim2) * num_classes + num_classes;

  const size_t batch_elems = static_cast<size_t>(batch_size);
  size_t static_buffers = param_elements;
  static_buffers += batch_elems * static_cast<size_t>(input_dim);
  static_buffers += batch_elems * static_cast<size_t>(num_classes);
  static_buffers +=
      static_cast<size_t>(eval_batch) * static_cast<size_t>(input_dim);

  const auto activation_block = [](size_t batch, int out_dim) {
    return batch * static_cast<size_t>(out_dim) * 3ULL;
  };

  const size_t forward_train =
      activation_block(batch_elems, hidden_dim1) +
      activation_block(batch_elems, hidden_dim2) +
      batch_elems * static_cast<size_t>(num_classes) * 2ULL;

  const size_t loss_buffers =
      batch_elems * static_cast<size_t>(num_classes) * 6ULL + 2048ULL;
  const size_t train_steps =
      static_cast<size_t>(epochs) * static_cast<size_t>(steps_per_epoch);
  const size_t train_hint = (forward_train + loss_buffers) * train_steps;

  const size_t eval_batch_sz = static_cast<size_t>(eval_batch);
  const size_t forward_eval =
    activation_block(eval_batch_sz, hidden_dim1) +
    activation_block(eval_batch_sz, hidden_dim2) +
    eval_batch_sz * static_cast<size_t>(num_classes) * 2ULL;

  const size_t val_iters =
      val_count > 0 ? static_cast<size_t>((val_count + eval_batch - 1) /
                                          eval_batch)
                    : 0ULL;
  const size_t test_iters =
      test_total > 0 ? static_cast<size_t>((test_total + eval_batch - 1) /
                                           eval_batch)
                     : 0ULL;
  const size_t eval_hint = (val_iters + test_iters) * forward_eval;

  size_t approx_hint = static_buffers + train_hint + eval_hint;
  approx_hint += approx_hint / 4ULL + 16384ULL;
  store.reserve(approx_hint);

  nn::Sequential model;
  model.emplace_back<nn::Linear>(input_dim, hidden_dim1, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim1, hidden_dim2, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim2, num_classes, store);

  auto params = model.params();

  Tensor batch_X = store.tensor({batch_size, input_dim});
  Tensor batch_y =
      store.tensor({batch_size, num_classes}, TensorInit::ZeroData);

  std::vector<float> epoch_losses;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_loss = 0.0f;
    for (int step = 0; step < steps_per_epoch; ++step) {
      store.zero_grad();
      batch_y.fill(0.0f);

      for (int i = 0; i < batch_size; ++i) {
        int idx = rand() % train_count;
        float *dst = batch_X.data() + i * input_dim;
        const auto &sample = mnist.data.train_data[idx];
        std::copy(sample.begin(), sample.end(), dst);
        int label = static_cast<int>(mnist.data.train_labels[idx]);
        fill_one_hot(batch_y, i, label);
      }

      Tensor logits = model(batch_X, store);
      Tensor loss = nn::bce_with_logits_loss(logits, batch_y, store);
      epoch_loss += loss.data()[0];

      store.backward(loss);
      nn::sgd_step(params, lr);
      store.clear_tape();
    }
    float avg_loss = epoch_loss / static_cast<float>(steps_per_epoch);
    epoch_losses.push_back(avg_loss);
    cout << "Epoch: " << epoch << " Avg Loss: " << avg_loss << endl;
  }

  cout << "Final training loss: "
       << (epoch_losses.empty() ? 0.0f : epoch_losses.back()) << endl;

  Tensor eval_X = store.tensor({eval_batch, input_dim});

  int correct = 0;
  int total = 0;
  int start_idx = train_count;
  int end_idx = train_count + val_count;
  end_idx = std::min(end_idx, total_samples);
  for (int idx = start_idx; idx < end_idx; idx += eval_batch) {
    int current_batch = std::min(eval_batch, end_idx - idx);
    for (int i = 0; i < current_batch; ++i) {
      float *dst = eval_X.data() + i * input_dim;
      const auto &sample = mnist.data.train_data[idx + i];
      std::copy(sample.begin(), sample.end(), dst);
    }
    Tensor logits = model(eval_X, store);
    const float *logits_ptr = logits.data();
    for (int i = 0; i < current_batch; ++i) {
      int predicted =
          argmax_from_logits(logits_ptr + i * num_classes, num_classes);
      int label = static_cast<int>(mnist.data.train_labels[idx + i]);
      if (predicted == label) ++correct;
      ++total;
    }
    store.clear_tape();
  }
  float val_accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;
  cout << "Validation accuracy (" << total << " samples): " << val_accuracy
       << endl;

  const auto &test_data = mnist.data.test_data;
  const auto &test_labels = mnist.data.test_labels;
  correct = 0;
  total = 0;
  for (int idx = 0; idx < test_total; idx += eval_batch) {
    int current_batch = std::min(eval_batch, test_total - idx);
    for (int i = 0; i < current_batch; ++i) {
      float *dst = eval_X.data() + i * input_dim;
      const auto &sample = test_data[idx + i];
      std::copy(sample.begin(), sample.end(), dst);
    }
    Tensor logits = model(eval_X, store);
    const float *logits_ptr = logits.data();
    for (int i = 0; i < current_batch; ++i) {
      int predicted =
          argmax_from_logits(logits_ptr + i * num_classes, num_classes);
      int label = static_cast<int>(test_labels[idx + i]);
      if (predicted == label) ++correct;
      ++total;
    }
    store.clear_tape();
  }
  float test_accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;
  cout << "Test accuracy (" << total << " samples): " << test_accuracy << endl;

  store.print_stats();
}
