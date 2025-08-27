#include "xormodel_tensors.hpp"

#include "nn.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace
{

struct Batch
{
    std::vector<std::array<float, 2>> x;
    std::vector<float> y;
};

float rand01()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void fill_tensor(Tensor &t, const std::vector<std::array<float, 2>> &rows)
{
    float *p = t.data();
    int N = t.shape[0];
    int D = t.shape[1];
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < D; ++j)
        {
            p[i * D + j] = rows[i][j];
        }
    }
}

void fill_tensor(Tensor &t, const std::vector<float> &vals)
{
    float *p = t.data();
    int N = t.shape[0];
    for (int i = 0; i < N; ++i)
        p[i] = vals[i];
}

Tensor mse_loss(const Tensor &pred, const Tensor &target, ParameterStore &store)
{
    auto diff = sub(pred, target, store);
    auto sq = mul(diff, diff, store);
    auto s = sum(sq, store);
    Tensor scale = store.tensor({1});
    scale.data()[0] = 1.0f / static_cast<float>(pred.shape[0]);
    return mul(s, scale, store);
}

} // namespace

// Define a small PyTorch-like Module inline for XOR
namespace nn
{
struct XORNet : public Module
{
    Linear l1;
    Tanh act;
    Linear l2;
    XORNet(int in_features, int hidden, int out_features, ParameterStore &store)
        : l1(in_features, hidden, store), act(), l2(hidden, out_features, store)
    {
    }
    Tensor forward(const Tensor &x, ParameterStore &store) override
    {
        auto h = l1(x, store);
        h = act(h, store);
        return l2(h, store);
    }
    std::vector<Tensor> params() override
    {
        auto p1 = l1.params();
        auto p2 = l2.params();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }
};
} // namespace nn

void XORWithTensors()
{
    using std::cout;
    using std::endl;
    srand(42);

    ParameterStore store;
    const int H = 16;      // hidden size
    const int N = 1024;    // dataset size
    const int B = 64;      // batch size
    const int EPOCHS = 80; // modest loops to validate
    const float LR = 0.3f;

    // Model class derived from Module with explicit forward
    nn::XORNet model(2, H, 1, store);
    auto params = model.params();

    // Dataset
    std::vector<std::array<float, 2>> X(N);
    std::vector<float> Y(N);
    // Build a simple XOR dataset from 4 corners, repeated
    std::array<std::array<float, 2>, 4> corners{{{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}}};
    std::array<float, 4> labels{{0.f, 1.f, 1.f, 0.f}};
    for (int i = 0; i < N; ++i)
    {
        X[i] = corners[i % 4];
        Y[i] = labels[i % 4];
    }

    auto forward = [&](const Tensor &Xb)
    { return model(Xb, store); };

    auto accuracy_on_sample = [&](int samples = 256)
    {
        store.clear_tape();
        int correct = 0;
        int total = samples;
        Tensor Xb = store.tensor({samples, 2});
        for (int i = 0; i < samples; ++i)
        {
            float a = rand01();
            float b = rand01();
            Xb.data()[i * 2 + 0] = a;
            Xb.data()[i * 2 + 1] = b;
        }
        auto yhat = forward(Xb); // [samples,1]
        for (int i = 0; i < samples; ++i)
        {
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
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        store.zero_grad();
        store.clear_tape();

        // sample batch
        Batch batch;
        batch.x.resize(B);
        batch.y.resize(B);
        for (int i = 0; i < B; ++i)
        {
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
        auto loss = nn::bce_with_logits_loss(logits, yb, store);

        store.backward(loss);
        nn::sgd_step(params, LR);
        store.clear_tape();

        if (epoch % 5 == 0 || epoch == EPOCHS - 1)
        {
            float L = loss.data()[0];
            // batch acc
            int bcorrect = 0;
            for (int i = 0; i < B; ++i)
            {
                bcorrect += (probs.data()[i] > 0.5f) == (yb.data()[i] > 0.5f);
            }
            float bacc = static_cast<float>(bcorrect) / B;
            // evaluate accuracy using probs
            store.clear_tape();
            int samples = 256;
            Tensor Xv = store.tensor({samples, 2});
            for (int i = 0; i < samples; ++i)
            {
                float a = rand01();
                float b = rand01();
                Xv.data()[i * 2 + 0] = a;
                Xv.data()[i * 2 + 1] = b;
            }
            auto pv = sigmoid(forward(Xv), store);
            int correct = 0;
            for (int i = 0; i < samples; ++i)
            {
                float a = Xv.data()[i * 2 + 0], b = Xv.data()[i * 2 + 1];
                float ytrue = ((a > 0.5f) ^ (b > 0.5f)) ? 1.0f : 0.0f;
                float ypred = pv.data()[i] > 0.5f ? 1.0f : 0.0f;
                correct += (ypred == ytrue);
            }
            float acc = static_cast<float>(correct) / samples;
            cout << "Epoch " << epoch << "\tLoss: " << L << "\tBatchAcc: " << bacc << "\tAcc: " << acc << std::endl;
        }
    }

    // Final accuracy
    store.clear_tape();
    int samples = 512;
    Tensor Xv = store.tensor({samples, 2});
    for (int i = 0; i < samples; ++i)
    {
        float a = rand01();
        float b = rand01();
        Xv.data()[i * 2] = a;
        Xv.data()[i * 2 + 1] = b;
    }
    auto pv = sigmoid(forward(Xv), store);
    int correct = 0;
    for (int i = 0; i < samples; ++i)
    {
        float a = Xv.data()[i * 2], b = Xv.data()[i * 2 + 1];
        float ytrue = ((a > 0.5f) ^ (b > 0.5f)) ? 1.0f : 0.0f;
        float ypred = pv.data()[i] > 0.5f ? 1.0f : 0.0f;
        correct += (ypred == ytrue);
    }
    cout << "Final accuracy: " << static_cast<float>(correct) / samples << endl;
}
