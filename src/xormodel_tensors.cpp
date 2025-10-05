#include "xormodel_tensors.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "nn.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace {

struct Batch {
  std::vector<std::array<float, 2>> x;
  std::vector<float> y;
};

void fill_tensor(Tensor &t, const std::vector<std::array<float, 2>> &rows) {
  float *p = t.data();
  int N = t.shape[0];
  int D = t.shape[1];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      p[i * D + j] = rows[i][j];
    }
  }
}

void fill_tensor(Tensor &t, const std::vector<float> &vals) {
  float *p = t.data();
  for (size_t i = 0; i < t.numel; ++i) p[i] = vals[i];
}

float xor_label(float a, float b) {
  return ((a > 0.5f) ^ (b > 0.5f)) ? 1.0f : 0.0f;
}

Batch sample_batch(const std::vector<std::array<float, 2>> &x,
                   const std::vector<float> &y, int batch_size) {
  Batch batch;
  batch.x.reserve(batch_size);
  batch.y.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    int idx = rand() % x.size();
    batch.x.push_back(x[idx]);
    batch.y.push_back(y[idx]);
  }
  return batch;
}

Tensor mse_loss(const Tensor &predicted, const Tensor &expected,
                ParameterStore &store) {
  auto diff = sub(predicted, expected, store);
  auto diff_sq = mul(diff, diff, store);
  auto total = sum(diff_sq, store);
  Tensor scale = store.tensor({1});
  scale.data()[0] = 1.0f / static_cast<float>(predicted.numel);
  return mul(total, scale, store);
}

}  // namespace


void XORWithTensors() {
  using std::cout;
  using std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  srand(42);

  ParameterStore store;
  store.enable_stats(true);
  constexpr int input_dim = 2;
  constexpr int hidden_dim1 = 10;
  constexpr int hidden_dim2 = 5;
  constexpr int output_dim = 1;
  constexpr int dataset_size = 100'000;
  constexpr int batch_size = 64;
  constexpr int epochs = 10;
  constexpr int trace_every = 1;
  constexpr float lr = 0.01f;

  nn::Sequential model;
  model.emplace_back<nn::Linear>(input_dim, hidden_dim1, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim1, hidden_dim2, store);
  model.emplace_back<nn::Relu>();
  model.emplace_back<nn::Linear>(hidden_dim2, output_dim, store);
  model.emplace_back<nn::Relu>();
  auto params = model.params();

  size_t total_param_count = 0;
  for (const auto &p : params) total_param_count += p.numel;
  cout << "Total params: " << total_param_count << endl;

  std::vector<std::array<float, 2>> x_all;
  x_all.reserve(dataset_size);
  std::vector<float> y_all;
  y_all.reserve(dataset_size);
  for (int i = 0; i < dataset_size; ++i) {
    std::array<float, 2> sample;
    sample[0] = get_random_float(0.0f, 1.0f);
    sample[1] = get_random_float(0.0f, 1.0f);
    x_all.push_back(sample);
    y_all.push_back(xor_label(sample[0], sample[1]));
  }

  constexpr float train_fraction = 0.8f;
  int train_size = static_cast<int>(train_fraction * dataset_size);
  std::vector<std::array<float, 2>> x_train(x_all.begin(),
                                            x_all.begin() + train_size);
  std::vector<float> y_train(y_all.begin(), y_all.begin() + train_size);
  std::vector<std::array<float, 2>> x_val(x_all.begin() + train_size,
                                          x_all.end());

  cout << "Total dataset size: " << dataset_size << endl;
  cout << "Batch size: " << batch_size << endl;
  cout << "Total epochs: " << epochs << endl;

  Tensor batch_X = store.tensor({batch_size, input_dim});
  Tensor batch_y = store.tensor({batch_size, output_dim});

  std::vector<float> losses;
  std::vector<std::vector<float>> y_val_epoch;
  std::vector<float> val_accuracy;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    store.zero_grad();
    store.clear_tape();

    Batch batch = sample_batch(x_train, y_train, batch_size);
    fill_tensor(batch_X, batch.x);
    fill_tensor(batch_y, batch.y);

    Tensor preds = model(batch_X, store);
    Tensor loss = mse_loss(preds, batch_y, store);
    float loss_value = loss.data()[0];
    losses.push_back(loss_value);

    store.backward(loss);
    nn::sgd_step(params, lr);
    store.clear_tape();

    if (epoch % trace_every == 0) {
      int val_size = static_cast<int>(x_val.size());
      Tensor Xv = store.tensor({val_size, input_dim});
      fill_tensor(Xv, x_val);
      Tensor logits_val = model(Xv, store);
      const float *logits_ptr = logits_val.data();

      int correct = 0;
      std::vector<float> y_val;
      y_val.reserve(val_size);
      for (int i = 0; i < val_size; ++i) {
        float out = logits_ptr[i];
        float out_thresh = out > 0.5f ? 1.0f : 0.0f;
        y_val.push_back(out_thresh);
        float label = xor_label(x_val[i][0], x_val[i][1]);
        if (out_thresh == label) {
          correct++;
        }
      }
      float accuracy = val_size > 0 ? static_cast<float>(correct) / val_size
                                    : 0.0f;
      cout << "Epoch: " << epoch << " Loss: " << loss_value
           << " Accuracy: " << accuracy << endl;
      val_accuracy.push_back(accuracy);
      y_val_epoch.push_back(std::move(y_val));
      store.clear_tape();
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto end_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  cout << "Time taken: " << end_ms << " ms" << endl;
  cout << "Time take per epoch per batch: "
       << static_cast<float>(end_ms) / (epochs * batch_size) << " ms" << endl;

  json losses_json = losses;
  dumpJson(losses_json, "data/losses.json");

  std::vector<float> param_values;
  for (const auto &param : params) {
    const float *p = param.data();
    param_values.insert(param_values.end(), p, p + param.numel);
  }
  json params_json = param_values;
  dumpJson(params_json, "data/params.json");

  std::vector<std::vector<float>> x_val_float;
  x_val_float.reserve(x_val.size());
  for (const auto &x : x_val) {
    x_val_float.push_back({x[0], x[1]});
  }
  json val_json = json{{"x", x_val_float},
                       {"y_across_epochs", y_val_epoch}};
  dumpJson(val_json, "data/xor_val.json");

  if (!val_accuracy.empty()) {
    cout << "Val accuracy in the end: " << val_accuracy.back() << endl;
  }
  if (!losses.empty()) {
    cout << "Loss in the end: " << losses.back() << endl;
  }

  const auto &stats = store.get_stats();
  const double tensor_avg_ms =
      stats.tensor_zero_calls
          ? stats.tensor_zero_ms / static_cast<double>(stats.tensor_zero_calls)
          : 0.0;
  const double zero_grad_avg_ms =
      stats.zero_grad_calls
          ? stats.zero_grad_ms / static_cast<double>(stats.zero_grad_calls)
          : 0.0;
  const double tensor_bytes =
      static_cast<double>(stats.tensor_zero_elems) * sizeof(float);
  const double zero_grad_bytes =
      static_cast<double>(stats.zero_grad_elems) * sizeof(float);
  const double tensor_mb = tensor_bytes / (1024.0 * 1024.0);
  const double zero_grad_mb = zero_grad_bytes / (1024.0 * 1024.0);

  cout << "ParameterStore zeroing stats:" << endl;
  cout << "  tensor() zero fills: " << stats.tensor_zero_calls
       << " calls, elements zeroed: " << stats.tensor_zero_elems
       << ", bytes zeroed: " << tensor_bytes << " (" << tensor_mb << " MB)"
       << ", total ms: " << stats.tensor_zero_ms
       << ", avg ms/call: " << tensor_avg_ms << endl;
  cout << "  zero_grad(): " << stats.zero_grad_calls
       << " calls, elements zeroed: " << stats.zero_grad_elems
       << ", bytes zeroed: " << zero_grad_bytes << " (" << zero_grad_mb
       << " MB)"
       << ", total ms: " << stats.zero_grad_ms
       << ", avg ms/call: " << zero_grad_avg_ms << endl;
}
