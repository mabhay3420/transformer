#include "mnist.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "dataloader.hpp"
#include "learning_rate.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include "utils.hpp"

void MnistDnnPT() {
  using std::cout;
  using std::endl;

  srand(42);

  const int MAX_TRAIN_SAMPLES = INT_MAX;
  const int MAX_TEST_SAMPLES = INT_MAX;
  const int default_epochs = 50;
  const int default_hidden1 = 512;
  const int default_hidden2 = 128;
  const int hidden_dim1 = getenv_int("MNIST_HIDDEN_DIM1", default_hidden1);
  const int hidden_dim2 = getenv_int("MNIST_HIDDEN_DIM2", default_hidden2);
  const int num_classes = 10;
  const int batch_size = std::max(1, getenv_int("MNIST_BATCH_SIZE", 128));
  const int eval_batch =
      std::max(1, getenv_int("MNIST_EVAL_BATCH_SIZE", batch_size));
  const int epochs = std::max(1, getenv_int("MNIST_EPOCHS", default_epochs));

  const auto dim_lr_scale = [](int dim, int baseline) {
    if (dim <= 0 || baseline <= 0) return 1.0f;
    const float denom = static_cast<float>(std::max(dim, baseline));
    return static_cast<float>(baseline) / denom;
  };
  const float base_lr = 0.001f;
  const float scaled_lr =
      base_lr * std::min(dim_lr_scale(hidden_dim1, default_hidden1),
                         dim_lr_scale(hidden_dim2, default_hidden2));
  const float lr = getenv_float("MNIST_LR", scaled_lr);

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

  cout << "Hyperparameters: hidden_dim1=" << hidden_dim1
       << ", hidden_dim2=" << hidden_dim2 << ", batch_size=" << batch_size
       << ", eval_batch=" << eval_batch << ", epochs=" << epochs
       << ", lr=" << lr << endl;

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

  const size_t eval_batch_sz = static_cast<size_t>(eval_batch);
  const size_t forward_eval =
      activation_block(eval_batch_sz, hidden_dim1) +
      activation_block(eval_batch_sz, hidden_dim2) +
      eval_batch_sz * static_cast<size_t>(num_classes) * 2ULL;

  const size_t per_step_scratch = forward_train + loss_buffers;
  const size_t per_eval_scratch = forward_eval;
  const size_t reserve_hint =
      static_buffers + per_step_scratch + per_eval_scratch + 16384ULL;
  store.reserve(reserve_hint);

  nn::Sequential model;
  model.emplace_back<nn::Linear>(input_dim, hidden_dim1, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim1, hidden_dim2, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim2, num_classes, store);

  auto params = model.params();

  const int lr_cliff = std::max(1, (steps_per_epoch * epochs) / 5);
  StepLRScheduler scheduler(lr, lr_cliff, 0.5f);
  optim::AdamW optimizer(params, scheduler, 0.9f, 0.999f, 1e-4f);

  Tensor batch_X = store.tensor({batch_size, input_dim});
  Tensor batch_y =
      store.tensor({batch_size, num_classes}, TensorInit::ZeroData);
  Tensor eval_X = store.tensor({eval_batch, input_dim});

  const size_t scratch_mark = store.mark();
  const auto reset_scratch = [&]() {
    store.reset(scratch_mark);
    store.clear_tape();
  };
  reset_scratch();

  std::vector<float> epoch_losses;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_loss = 0.0f;
    for (int step = 0; step < steps_per_epoch; ++step) {
      reset_scratch();
      optimizer.zero_grad();
      batch_y.fill(0.0f);

      for (int i = 0; i < batch_size; ++i) {
        int idx = rand() % train_count;
        float* dst = batch_X.data() + i * input_dim;
        const auto& sample = mnist.data.train_data[idx];
        std::copy(sample.begin(), sample.end(), dst);
        int label = static_cast<int>(mnist.data.train_labels[idx]);
        fill_one_hot(batch_y, i, label);
      }

      Tensor logits = model(batch_X, store);
      Tensor loss = nn::bce_with_logits_loss(logits, batch_y, store);
      epoch_loss += loss.data()[0];

      store.backward(loss);
      optimizer.step();
    }
    float avg_loss = epoch_loss / static_cast<float>(steps_per_epoch);
    epoch_losses.push_back(avg_loss);
    cout << "Epoch: " << epoch << " Avg Loss: " << avg_loss << endl;
  }

  reset_scratch();

  cout << "Final training loss: "
       << (epoch_losses.empty() ? 0.0f : epoch_losses.back()) << endl;

  int correct = 0;
  int total = 0;
  int start_idx = train_count;
  int end_idx = train_count + val_count;
  end_idx = std::min(end_idx, total_samples);
  for (int idx = start_idx; idx < end_idx; idx += eval_batch) {
    reset_scratch();
    int current_batch = std::min(eval_batch, end_idx - idx);
    for (int i = 0; i < current_batch; ++i) {
      float* dst = eval_X.data() + i * input_dim;
      const auto& sample = mnist.data.train_data[idx + i];
      std::copy(sample.begin(), sample.end(), dst);
    }
    Tensor logits = model(eval_X, store);
    const float* logits_ptr = logits.data();
    for (int i = 0; i < current_batch; ++i) {
      int predicted =
          argmax_from_logits(logits_ptr + i * num_classes, num_classes);
      int label = static_cast<int>(mnist.data.train_labels[idx + i]);
      if (predicted == label) ++correct;
      ++total;
    }
  }
  float val_accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;
  cout << "Validation accuracy (" << total << " samples): " << val_accuracy
       << endl;

  const auto& test_data = mnist.data.test_data;
  const auto& test_labels = mnist.data.test_labels;
  correct = 0;
  total = 0;
  for (int idx = 0; idx < test_total; idx += eval_batch) {
    reset_scratch();
    int current_batch = std::min(eval_batch, test_total - idx);
    for (int i = 0; i < current_batch; ++i) {
      float* dst = eval_X.data() + i * input_dim;
      const auto& sample = test_data[idx + i];
      std::copy(sample.begin(), sample.end(), dst);
    }
    Tensor logits = model(eval_X, store);
    const float* logits_ptr = logits.data();
    for (int i = 0; i < current_batch; ++i) {
      int predicted =
          argmax_from_logits(logits_ptr + i * num_classes, num_classes);
      int label = static_cast<int>(test_labels[idx + i]);
      if (predicted == label) ++correct;
      ++total;
    }
  }
  float test_accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;
  cout << "Test accuracy (" << total << " samples): " << test_accuracy << endl;

  store.print_stats();
}
