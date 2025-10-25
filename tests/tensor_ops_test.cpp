#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "learning_rate.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include "utils.hpp"

static void fill_vec(float* p, const std::vector<float>& vals) {
  std::copy(vals.begin(), vals.end(), p);
}

TEST(ParameterStore, ReserveHint) {
  ParameterStore ps;
  EXPECT_EQ(ps.size(), 0u);
  EXPECT_EQ(ps.capacity_count(), 0u);

  ps.reserve(128);
  EXPECT_GE(ps.capacity_count(), 128u);
  EXPECT_EQ(ps.size(), 0u);

  auto t = ps.tensor({16});
  EXPECT_EQ(t.numel, 16u);
  EXPECT_EQ(ps.size(), 16u);
  EXPECT_GE(ps.capacity_count(), 128u);

  const size_t prev_capacity = ps.capacity_count();
  ps.reserve(ps.size() + 256u);
  EXPECT_EQ(ps.size(), 16u);
  EXPECT_GE(ps.capacity_count(), ps.size() + 256u);

  ps.reserve(8u);
  EXPECT_GE(ps.capacity_count(), prev_capacity);
}

TEST(ParameterStore, ResetReuse) {
  ParameterStore ps;
  auto persistent = ps.tensor({4}, TensorInit::ZeroData);
  fill_vec(persistent.data(), {1.f, 2.f, 3.f, 4.f});

  size_t mark = ps.mark();
  auto scratch = ps.tensor({2});
  const size_t scratch_offset = scratch.offset;

  ps.reset(mark);
  auto reused = ps.tensor({2});
  EXPECT_EQ(reused.offset, scratch_offset);
  EXPECT_FLOAT_EQ(persistent.data()[0], 1.f);
  EXPECT_FLOAT_EQ(persistent.data()[3], 4.f);
}

TEST(TensorOps, AddBackward) {
  ParameterStore ps;
  ps.clear_tape();
  auto a = ps.tensor({2, 3});
  auto b = ps.tensor({2, 3});
  fill_vec(a.data(), {1, 2, 3, 4, 5, 6});
  fill_vec(b.data(), {6, 5, 4, 3, 2, 1});
  auto c = add(a, b, ps);
  auto s = sum(c, ps);
  ps.zero_grad();
  ps.backward(s);
  for (size_t i = 0; i < a.numel; ++i) {
    EXPECT_FLOAT_EQ(a.grad()[i], 1.0f);
  }
  for (size_t i = 0; i < b.numel; ++i) {
    EXPECT_FLOAT_EQ(b.grad()[i], 1.0f);
  }
}

TEST(TensorOps, MulBackward) {
  ParameterStore ps;
  ps.clear_tape();
  auto a = ps.tensor({2, 2});
  auto b = ps.tensor({2, 2});
  fill_vec(a.data(), {1, 2, 3, 4});
  fill_vec(b.data(), {5, 6, 7, 8});
  auto c = mul(a, b, ps);
  auto s = sum(c, ps);
  ps.zero_grad();
  ps.backward(s);
  for (size_t i = 0; i < a.numel; ++i)
    EXPECT_FLOAT_EQ(a.grad()[i], b.data()[i]);
  for (size_t i = 0; i < b.numel; ++i)
    EXPECT_FLOAT_EQ(b.grad()[i], a.data()[i]);
}

TEST(TensorOps, ReluBackward) {
  ParameterStore ps;
  ps.clear_tape();
  auto x = ps.tensor({4});
  fill_vec(x.data(), {-1.f, 0.2f, 3.f, -0.5f});
  auto y = relu(x, ps);
  auto s = sum(y, ps);
  ps.zero_grad();
  ps.backward(s);
  std::vector<float> expect = {0.f, 1.f, 1.f, 0.f};
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(x.grad()[i], expect[i]);
}

TEST(TensorOps, SumBackward) {
  ParameterStore ps;
  ps.clear_tape();
  auto x = ps.tensor({3});
  fill_vec(x.data(), {1.f, 2.f, 3.f});
  auto s = sum(x, ps);
  ps.zero_grad();
  ps.backward(s);
  for (int i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(x.grad()[i], 1.f);
}

TEST(TensorOps, MatmulBackward) {
  ParameterStore ps;
  ps.clear_tape();
  // A[2,2], B[2,2]
  auto A = ps.tensor({2, 2});
  auto B = ps.tensor({2, 2});
  fill_vec(A.data(), {1.f, 2.f, 3.f, 4.f});  // rows: [1,2], [3,4]
  fill_vec(B.data(), {5.f, 6.f, 7.f, 8.f});  // rows: [5,6], [7,8]
  auto C = matmul(A, B, ps);
  auto s = sum(C, ps);
  ps.zero_grad();
  ps.backward(s);
  // dA[m,k] = sum_n B[k,n] => row sums of B rows
  float r0 = B.data()[0] + B.data()[1];  // 5+6=11
  float r1 = B.data()[2] + B.data()[3];  // 7+8=15
  EXPECT_FLOAT_EQ(A.grad()[0], r0);
  EXPECT_FLOAT_EQ(A.grad()[1], r1);
  EXPECT_FLOAT_EQ(A.grad()[2], r0);
  EXPECT_FLOAT_EQ(A.grad()[3], r1);
  // dB[k,n] = sum_m A[m,k] => column sums of A columns
  float c0 = A.data()[0] + A.data()[2];  // 1+3=4
  float c1 = A.data()[1] + A.data()[3];  // 2+4=6
  // row 0 of B grad all = c0; row 1 all = c1
  EXPECT_FLOAT_EQ(B.grad()[0], c0);
  EXPECT_FLOAT_EQ(B.grad()[1], c0);
  EXPECT_FLOAT_EQ(B.grad()[2], c1);
  EXPECT_FLOAT_EQ(B.grad()[3], c1);
}

TEST(TensorOps, AddRowwiseBackward) {
  ParameterStore ps;
  ps.clear_tape();
  auto X = ps.tensor({3, 2});
  auto b = ps.tensor({2});
  fill_vec(X.data(), {1, 2, 3, 4, 5, 6});
  fill_vec(b.data(), {0.5f, -1.0f});
  auto Y = add_rowwise(X, b, ps);
  auto s = sum(Y, ps);
  ps.zero_grad();
  ps.backward(s);
  for (int i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(X.grad()[i], 1.f);
  // grad wrt b: [N, N]
  EXPECT_FLOAT_EQ(b.grad()[0], 3.f);
  EXPECT_FLOAT_EQ(b.grad()[1], 3.f);
}

TEST(TensorOps, SigmoidGradMatchesAnalytic) {
  ParameterStore ps;
  ps.clear_tape();
  auto x = ps.tensor({3});
  fill_vec(x.data(), {-1.f, 0.0f, 2.f});
  auto y = sigmoid(x, ps);
  auto s = sum(y, ps);
  ps.zero_grad();
  ps.backward(s);
  for (int i = 0; i < 3; ++i) {
    float sig = 1.0f / (1.0f + std::exp(-x.data()[i]));
    float expected = sig * (1.0f - sig);
    EXPECT_NEAR(x.grad()[i], expected, 1e-5);
  }
}

TEST(TensorOps, LogGrad) {
  ParameterStore ps;
  ps.clear_tape();
  auto x = ps.tensor({3});
  fill_vec(x.data(), {0.5f, 2.0f, 4.0f});
  auto y = vlog(x, ps);
  auto s = sum(y, ps);
  ps.zero_grad();
  ps.backward(s);
  EXPECT_FLOAT_EQ(x.grad()[0], 1.0f / 0.5f);
  EXPECT_FLOAT_EQ(x.grad()[1], 1.0f / 2.0f);
  EXPECT_FLOAT_EQ(x.grad()[2], 1.0f / 4.0f);
}

TEST(NN, BCEWithLogitsGradMatchesAnalytic) {
  ParameterStore ps;
  ps.clear_tape();
  auto logits = ps.tensor({2, 1});
  fill_vec(logits.data(), {0.2f, -0.7f});
  auto targets = ps.tensor({2, 1});
  fill_vec(targets.data(), {1.0f, 0.0f});

  auto loss = nn::bce_with_logits_loss(logits, targets, ps);
  ps.zero_grad();
  ps.backward(loss);

  const float invN = 1.0f / static_cast<float>(targets.shape[0]);
  for (int i = 0; i < logits.shape[0]; ++i) {
    const float sig = 1.0f / (1.0f + std::exp(-logits.data()[i]));
    const float expected = (sig - targets.data()[i]) * invN;
    EXPECT_NEAR(logits.grad()[i], expected, 1e-5f);
  }
}

TEST(Optimizer, SGDBasicStep) {
  ParameterStore ps;
  auto param = ps.tensor({2}, TensorInit::ZeroData);
  float* data = param.data();
  data[0] = 1.0f;
  data[1] = -1.0f;
  float* grad = param.grad();
  grad[0] = 0.5f;
  grad[1] = -0.25f;

  ConstantLRScheduler scheduler(0.1f);
  optim::SGD optimizer({param}, scheduler);
  optimizer.step();

  EXPECT_FLOAT_EQ(data[0], 0.95f);
  EXPECT_FLOAT_EQ(data[1], -0.975f);
}

TEST(Optimizer, AdamWDecoupledWeightDecay) {
  ParameterStore ps;
  auto param = ps.tensor({1}, TensorInit::ZeroData);
  param.data()[0] = 2.0f;
  param.grad()[0] = 0.5f;

  ConstantLRScheduler scheduler(0.1f);
  optim::AdamW optimizer({param}, scheduler, 0.9f, 0.999f, 0.01f, true, false,
                         1e-8f);
  optimizer.step();

  const float expected_decay = 2.0f - 0.1f * 0.01f * 2.0f;
  const float expected_value = expected_decay - 0.1f;
  EXPECT_NEAR(param.data()[0], expected_value, 1e-6f);
}

TEST(TensorOps, TensorFillAndZeroGrad) {
  ParameterStore ps;
  auto t = ps.tensor({2, 3});
  t.fill(5.0f);
  for (size_t i = 0; i < t.numel; ++i) {
    EXPECT_FLOAT_EQ(t.data()[i], 5.0f);
  }
  // Set some grads
  t.grad()[0] = 1.0f;
  t.grad()[1] = 2.0f;
  t.zero_grad();
  for (size_t i = 0; i < t.numel; ++i) {
    EXPECT_FLOAT_EQ(t.grad()[i], 0.0f);
  }
}

TEST(TensorOps, ReluForward) {
  ParameterStore ps;
  auto x = ps.tensor({4});
  fill_vec(x.data(), {-1.0f, 0.0f, 2.0f, -0.5f});
  auto y = relu(x, ps);
  EXPECT_FLOAT_EQ(y.data()[0], 0.0f);
  EXPECT_FLOAT_EQ(y.data()[1], 0.0f);
  EXPECT_FLOAT_EQ(y.data()[2], 2.0f);
  EXPECT_FLOAT_EQ(y.data()[3], 0.0f);
}

TEST(TensorOps, SigmoidForward) {
  ParameterStore ps;
  auto x = ps.tensor({3});
  fill_vec(x.data(), {0.0f, 1.0f, -1.0f});
  auto y = sigmoid(x, ps);
  EXPECT_NEAR(y.data()[0], 0.5f, 1e-6);
  EXPECT_NEAR(y.data()[1], 1.0f / (1.0f + std::exp(-1.0f)), 1e-6);
  EXPECT_NEAR(y.data()[2], 1.0f / (1.0f + std::exp(1.0f)), 1e-6);
}

TEST(TensorOps, LogForward) {
  ParameterStore ps;
  auto x = ps.tensor({3});
  fill_vec(x.data(), {1.0f, std::exp(1.0f), std::exp(2.0f)});
  auto y = vlog(x, ps);
  EXPECT_NEAR(y.data()[0], 0.0f, 1e-6);
  EXPECT_NEAR(y.data()[1], 1.0f, 1e-6);
  EXPECT_NEAR(y.data()[2], 2.0f, 1e-6);
}

TEST(TensorOps, SoftmaxAndArgmax) {
  std::vector<float> logits = {1.0f, 2.0f, 0.5f};
  auto probs = softmax_from_logits(logits.data(), logits.size());
  EXPECT_NEAR(probs[0] + probs[1] + probs[2], 1.0f, 1e-6);
  int argmax_idx = argmax_from_logits(logits.data(), logits.size());
  EXPECT_EQ(argmax_idx, 1);
}

TEST(TensorOps, FillOneHot) {
  ParameterStore ps;
  auto t = ps.tensor({1, 3}, TensorInit::ZeroData);
  fill_one_hot(t, 0, 1);
  EXPECT_FLOAT_EQ(t.data()[0], 0.0f);
  EXPECT_FLOAT_EQ(t.data()[1], 1.0f);
  EXPECT_FLOAT_EQ(t.data()[2], 0.0f);
}

TEST(TensorOps, MatmulDifferentShapes) {
  ParameterStore ps;
  // 1x2 @ 2x3 = 1x3
  auto a = ps.tensor({1, 2});
  auto b = ps.tensor({2, 3});
  fill_vec(a.data(), {1.0f, 2.0f});
  fill_vec(b.data(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto c = matmul(a, b, ps);
  EXPECT_EQ(c.shape[0], 1);
  EXPECT_EQ(c.shape[1], 3);
  EXPECT_FLOAT_EQ(c.data()[0], 1 * 1 + 2 * 4);  // 1*1 + 2*4 = 9
  EXPECT_FLOAT_EQ(c.data()[1], 1 * 2 + 2 * 5);  // 12
  EXPECT_FLOAT_EQ(c.data()[2], 1 * 3 + 2 * 6);  // 15
}

TEST(TensorOps, MatmulInvalidShapes) {
  ParameterStore ps;
  auto a = ps.tensor({2, 3});
  auto b = ps.tensor({4, 5});
  EXPECT_THROW(matmul(a, b, ps), std::invalid_argument);
}

TEST(TensorOps, MatmulNon2D) {
  ParameterStore ps;
  auto a = ps.tensor({3});
  auto b = ps.tensor({3});
  EXPECT_THROW(matmul(a, b, ps), std::invalid_argument);
}

TEST(Scheduler, ConstantLRScheduler) {
  ConstantLRScheduler sched(0.01f);
  EXPECT_FLOAT_EQ(sched.get(), 0.01f);
  EXPECT_FLOAT_EQ(sched.get(), 0.01f);
}

TEST(Scheduler, StepLRScheduler) {
  StepLRScheduler sched(0.1f, 2, 0.5f);
  EXPECT_FLOAT_EQ(sched.get(), 0.1f);   // step 1
  EXPECT_FLOAT_EQ(sched.get(), 0.05f);  // step 2, reduce
  EXPECT_FLOAT_EQ(sched.get(), 0.05f);  // step 3
}

TEST(Scheduler, StepLRSchedulerHonorsLimit) {
  StepLRScheduler sched(0.1f, 1, 0.5f, 0.03f);
  EXPECT_FLOAT_EQ(sched.get(), 0.05f);
  EXPECT_FLOAT_EQ(sched.get(), 0.03f);
  EXPECT_FLOAT_EQ(sched.get(), 0.03f);
}

TEST(Scheduler, StepLRSchedulerRejectsInvalidParameters) {
  EXPECT_THROW(StepLRScheduler(0.1f, 0, 0.5f), std::invalid_argument);
  EXPECT_THROW(StepLRScheduler(0.1f, -3, 0.5f), std::invalid_argument);
  EXPECT_THROW(StepLRScheduler(0.1f, 4, 0.0f), std::invalid_argument);
  EXPECT_THROW(StepLRScheduler(0.1f, 4, -0.1f), std::invalid_argument);
}

TEST(Optimizer, AdamBasicStep) {
  ParameterStore ps;
  auto param = ps.tensor({1}, TensorInit::ZeroData);
  param.data()[0] = 1.0f;
  param.grad()[0] = 0.1f;

  ConstantLRScheduler scheduler(0.01f);
  optim::Adam optimizer({param}, scheduler);

  optimizer.step();
  // Approximate check, exact calculation is complex
  EXPECT_TRUE(param.data()[0] < 1.0f);  // Should decrease
}

TEST(ParameterStore, MultipleTensors) {
  ParameterStore ps;
  auto t1 = ps.tensor({2});
  auto t2 = ps.tensor({3});
  EXPECT_EQ(ps.size(), 5u);
  EXPECT_GE(ps.capacity_count(), 5u);
}

TEST(NN, LinearLayer) {
  ParameterStore ps;
  nn::Linear linear(4, 2, ps);
  auto input = ps.tensor({1, 4});
  fill_vec(input.data(), {1.0f, 2.0f, 3.0f, 4.0f});
  auto output = linear.forward(input, ps);
  EXPECT_EQ(output.shape[0], 1);
  EXPECT_EQ(output.shape[1], 2);
}

TEST(NN, SequentialModel) {
  ParameterStore ps;
  nn::Sequential model;
  model.emplace_back<nn::Linear>(2, 3, ps);
  model.emplace_back<nn::Relu>();
  auto input = ps.tensor({1, 2});
  fill_vec(input.data(), {1.0f, -1.0f});
  auto output = model(input, ps);
  EXPECT_EQ(output.shape[0], 1);
  EXPECT_EQ(output.shape[1], 3);
  // Relu should make negative zero
  EXPECT_TRUE(output.data()[0] >= 0.0f);
  EXPECT_FLOAT_EQ(output.data()[1], 0.0f);
  EXPECT_TRUE(output.data()[2] >= 0.0f);
}

TEST(NN, LinearDeterministicDefaultSeed) {
  ParameterStore ps1;
  ParameterStore ps2;
  nn::Linear linear1(2, 2, ps1);
  nn::Linear linear2(2, 2, ps2);
  auto params1 = linear1.params();
  auto params2 = linear2.params();
  ASSERT_EQ(params1.size(), params2.size());
  for (size_t p = 0; p < params1.size(); ++p) {
    ASSERT_EQ(params1[p].numel, params2[p].numel);
    for (size_t i = 0; i < params1[p].numel; ++i) {
      EXPECT_FLOAT_EQ(params1[p].data()[i], params2[p].data()[i]);
    }
  }
}
