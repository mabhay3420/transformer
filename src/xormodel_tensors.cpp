#include "xormodel_tensors.hpp"

#include "tensor.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct Batch {
  std::vector<std::array<float, 2>> x;
  std::vector<float> y;
};

float rand01() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }

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

Tensor mse_loss(const Tensor &pred, const Tensor &target, ParameterStore &store) {
  auto diff = sub(pred, target, store);
  auto sq = mul(diff, diff, store);
  auto s = sum(sq, store);
  Tensor scale = store.tensor({1});
  scale.data()[0] = 1.0f / static_cast<float>(pred.shape[0]);
  return mul(s, scale, store);
}

void sgd_step(const std::vector<Tensor> &params, float lr) {
  for (const auto &p : params) {
    float *w = p.store->data_buf.data() + p.offset;
    float *g = p.store->grad_buf.data() + p.offset;
    for (size_t i = 0; i < p.numel; ++i) {
      w[i] -= lr * g[i];
    }
  }
}

} // namespace

void XORWithTensors() {
  using std::cout;
  using std::endl;
  srand(42);

  ParameterStore store;
  const int H = 16;      // hidden size
  const int N = 1024;    // dataset size
  const int B = 64;      // batch size
  const int EPOCHS = 120; // few loops to validate convergence
  const float LR = 0.5f;

  // Model: 2 -> H -> 1 with ReLU + Sigmoid
  auto W1 = store.parameter({2, H}, 0.5f, 123);
  auto b1 = store.parameter({H}, 0.5f, 321);
  auto W2 = store.parameter({H, 1}, 0.5f, 456);
  auto b2 = store.parameter({1}, 0.5f, 654);
  std::vector<Tensor> params = {W1, b1, W2, b2};

  // Dataset
  std::vector<std::array<float, 2>> X(N);
  std::vector<float> Y(N);
  // Build a simple XOR dataset from 4 corners, repeated
  std::array<std::array<float,2>,4> corners{{ {0.f,0.f}, {0.f,1.f}, {1.f,0.f}, {1.f,1.f} }};
  std::array<float,4> labels{{0.f, 1.f, 1.f, 0.f}};
  for (int i = 0; i < N; ++i) { X[i] = corners[i % 4]; Y[i] = labels[i % 4]; }

  auto forward = [&](const Tensor &Xb) {
    auto h1 = matmul(Xb, W1, store);
    auto h1b = add_rowwise(h1, b1, store);
    auto h1a = vtanh(h1b, store);
    auto o = matmul(h1a, W2, store);
    auto ob = add_rowwise(o, b2, store); // logits
    return ob;
  };

  auto accuracy_on_sample = [&](int samples = 256) {
    store.clear_tape();
    int correct = 0;
    int total = samples;
    Tensor Xb = store.tensor({samples, 2});
    for (int i = 0; i < samples; ++i) {
      float a = rand01();
      float b = rand01();
      Xb.data()[i * 2 + 0] = a;
      Xb.data()[i * 2 + 1] = b;
    }
    auto yhat = forward(Xb); // [samples,1]
    for (int i = 0; i < samples; ++i) {
      float a = Xb.data()[i * 2 + 0];
      float b = Xb.data()[i * 2 + 1];
      float ytrue = ((a > 0.5f) ^ (b > 0.5f)) ? 1.0f : 0.0f;
      float pred = yhat.data()[i];
      float ypred = pred > 0.5f ? 1.0f : 0.0f;
      correct += (ypred == ytrue);
    }
    return static_cast<float>(correct) / total;
  };

  cout << "Training XOR with Tensor autograd..." << endl;
  for (int epoch = 0; epoch < EPOCHS; ++epoch) {
    store.zero_grad();
    store.clear_tape();

    // sample batch
    Batch batch;
    batch.x.resize(B);
    batch.y.resize(B);
    for (int i = 0; i < B; ++i) {
      int idx = rand() % N;
      batch.x[i] = X[idx];
      batch.y[i] = Y[idx];
    }

    Tensor Xb = store.tensor({B, 2});
    Tensor yb = store.tensor({B, 1});
    fill_tensor(Xb, batch.x);
    fill_tensor(yb, batch.y);

    auto logits = forward(Xb); // [B,1]
    auto probs = sigmoid(logits, store);
    // Binary cross-entropy loss
    Tensor ones = store.tensor({B, 1});
    std::fill(ones.data(), ones.data() + ones.numel, 1.0f);
    Tensor epsT = store.tensor({B, 1});
    std::fill(epsT.data(), epsT.data() + epsT.numel, 1e-6f);
    auto p_eps = add(probs, epsT, store);
    auto q = sub(ones, probs, store);
    auto q_eps = add(q, epsT, store);
    auto term1 = mul(yb, vlog(p_eps, store), store);
    auto one_minus_y = sub(ones, yb, store);
    auto term2 = mul(one_minus_y, vlog(q_eps, store), store);
    auto bce = add(term1, term2, store);
    auto s = sum(bce, store);
    Tensor scale = store.tensor({1});
    scale.data()[0] = -1.0f / static_cast<float>(B);
    auto loss = mul(s, scale, store);

    store.backward(loss);
    // debug grad norms
    auto gw1 = 0.0f; for (size_t i = 0; i < W1.numel; ++i) gw1 += std::fabs(W1.grad()[i]);
    auto gw2 = 0.0f; for (size_t i = 0; i < W2.numel; ++i) gw2 += std::fabs(W2.grad()[i]);
    auto gb1 = 0.0f; for (size_t i = 0; i < b1.numel; ++i) gb1 += std::fabs(b1.grad()[i]);
    auto gb2 = 0.0f; for (size_t i = 0; i < b2.numel; ++i) gb2 += std::fabs(b2.grad()[i]);
    sgd_step(params, LR);
    store.clear_tape();

    if (epoch % 5 == 0 || epoch == EPOCHS - 1) {
      float L = loss.data()[0];
      // batch acc
      int bcorrect = 0; for (int i=0;i<B;++i){ bcorrect += (probs.data()[i] > 0.5f) == (yb.data()[i] > 0.5f); }
      float bacc = static_cast<float>(bcorrect)/B;
      // evaluate accuracy using probs
      store.clear_tape();
      int samples = 256;
      Tensor Xv = store.tensor({samples, 2});
      for (int i = 0; i < samples; ++i) {
        float a = rand01(); float b = rand01();
        Xv.data()[i*2+0] = a; Xv.data()[i*2+1] = b;
      }
      auto pv = sigmoid(forward(Xv), store);
      int correct = 0;
      for (int i = 0; i < samples; ++i) {
        float a = Xv.data()[i*2+0], b = Xv.data()[i*2+1];
        float ytrue = ((a>0.5f)^(b>0.5f)) ? 1.0f:0.0f;
        float ypred = pv.data()[i] > 0.5f ? 1.0f : 0.0f;
        correct += (ypred == ytrue);
      }
      float acc = static_cast<float>(correct)/samples;
      cout << "Epoch " << epoch << "\tLoss: " << L << "\tBatchAcc: " << bacc << "\tAcc: " << acc
           << "\t|grad|: W1=" << gw1/W1.numel << ", b1=" << gb1/b1.numel
           << ", W2=" << gw2/W2.numel << ", b2=" << gb2/b2.numel << endl;
    }
  }

  // Final accuracy
  store.clear_tape();
  int samples = 512;
  Tensor Xv = store.tensor({samples, 2});
  for (int i = 0; i < samples; ++i) { float a = rand01(); float b = rand01(); Xv.data()[i*2] = a; Xv.data()[i*2+1] = b; }
  auto pv = sigmoid(forward(Xv), store);
  int correct = 0; for (int i = 0; i < samples; ++i) { float a = Xv.data()[i*2], b = Xv.data()[i*2+1]; float ytrue = ((a>0.5f)^(b>0.5f))?1.0f:0.0f; float ypred = pv.data()[i] > 0.5f ? 1.0f : 0.0f; correct += (ypred == ytrue);} 
  cout << "Final accuracy: " << static_cast<float>(correct)/samples << endl;
}
