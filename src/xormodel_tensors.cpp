#include "xormodel_tensors.hpp"

#include <array>
#include <iostream>
#include <random>
#include <vector>

#include "nn.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace {

struct Batch {
  std::vector<std::array<float, 2>> x;
  std::vector<float> y;
};

float rand01(std::mt19937 &rng) {
  static std::uniform_real_distribution<float> dist(0.f, 1.f);
  return dist(rng);
}

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
  int N = t.shape[0];
  for (int i = 0; i < N; ++i) p[i] = vals[i];
}

float xor_label(float a, float b) {
  return ((a > 0.5f) ^ (b > 0.5f)) ? 1.0f : 0.0f;
}

Batch sample_batch(const std::vector<std::array<float, 2>> &x,
                   const std::vector<float> &y, int batch_size,
                   std::mt19937 &rng) {
  Batch batch;
  batch.x.reserve(batch_size);
  batch.y.reserve(batch_size);
  std::uniform_int_distribution<int> dist(0, static_cast<int>(x.size()) - 1);
  for (int i = 0; i < batch_size; ++i) {
    int idx = dist(rng);
    batch.x.push_back(x[idx]);
    batch.y.push_back(y[idx]);
  }
  return batch;
}

float binary_accuracy(const Tensor &probabilities, const Tensor &targets) {
  int correct = 0;
  auto *p = probabilities.data();
  auto *t = targets.data();
  for (size_t i = 0; i < probabilities.numel; ++i) {
    correct += (p[i] > 0.5f) == (t[i] > 0.5f);
  }
  return static_cast<float>(correct) / probabilities.numel;
}

}  // namespace


void XORWithTensors() {
  using std::cout;
  using std::endl;
  std::mt19937 rng(42);

  ParameterStore store;
  constexpr int input_dim = 2;
  constexpr int hidden_dim = 16;
  constexpr int output_dim = 1;
  constexpr int dataset_size = 1024;
  constexpr int batch_size = 64;
  constexpr int epochs = 80;
  constexpr float lr = 0.3f;

  nn::Sequential model;
  model.emplace_back<nn::Linear>(input_dim, hidden_dim, store);
  model.emplace_back<nn::Tanh>();
  model.emplace_back<nn::Linear>(hidden_dim, output_dim, store);
  auto params = model.params();

  std::vector<std::array<float, 2>> features(dataset_size);
  std::vector<float> targets(dataset_size);
  const std::array<std::array<float, 2>, 4> corners{
      {{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}}};
  const std::array<float, 4> labels{{0.f, 1.f, 1.f, 0.f}};
  for (int i = 0; i < dataset_size; ++i) {
    features[i] = corners[i % 4];
    targets[i] = labels[i % 4];
  }

  cout << "Training XOR with Tensor autograd (PyTorch-style nn)..." << endl;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    store.zero_grad();
    store.clear_tape();

    Batch batch = sample_batch(features, targets, batch_size, rng);
    Tensor Xb = store.tensor({batch_size, input_dim});
    Tensor yb = store.tensor({batch_size, output_dim});
    fill_tensor(Xb, batch.x);
    fill_tensor(yb, batch.y);

    Tensor logits = model(Xb, store);
    Tensor loss = nn::bce_with_logits_loss(logits, yb, store);
    Tensor probs = sigmoid(logits, store);

    store.backward(loss);
    nn::sgd_step(params, lr);
    store.clear_tape();

    if (epoch % 5 == 0 || epoch == epochs - 1) {
      const float batch_loss = loss.data()[0];
      const float batch_acc = binary_accuracy(probs, yb);

      store.clear_tape();
      const int samples = 256;
      Tensor Xv = store.tensor({samples, input_dim});
      Tensor yv = store.tensor({samples, output_dim});
      for (int i = 0; i < samples; ++i) {
        float a = rand01(rng);
        float b = rand01(rng);
        Xv.data()[i * input_dim + 0] = a;
        Xv.data()[i * input_dim + 1] = b;
        yv.data()[i] = xor_label(a, b);
      }
      Tensor pv = sigmoid(model(Xv, store), store);
      const float acc = binary_accuracy(pv, yv);

      cout << "Epoch " << epoch << "\tLoss: " << batch_loss
           << "\tBatchAcc: " << batch_acc << "\tAcc: " << acc << endl;
    }
  }

  store.clear_tape();
  const int samples = 512;
  Tensor Xv = store.tensor({samples, input_dim});
  Tensor yv = store.tensor({samples, output_dim});
  for (int i = 0; i < samples; ++i) {
    float a = rand01(rng);
    float b = rand01(rng);
    Xv.data()[i * input_dim + 0] = a;
    Xv.data()[i * input_dim + 1] = b;
    yv.data()[i] = xor_label(a, b);
  }
  Tensor pv = sigmoid(model(Xv, store), store);
  cout << "Final accuracy: " << binary_accuracy(pv, yv) << endl;
}
