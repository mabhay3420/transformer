#include <gtest/gtest.h>

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

static void fill_vec(float *p, const std::vector<float> &vals)
{
    std::copy(vals.begin(), vals.end(), p);
}

TEST(TensorOps, AddBackward)
{
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
    for (size_t i = 0; i < a.numel; ++i)
    {
        EXPECT_FLOAT_EQ(a.grad()[i], 1.0f);
    }
    for (size_t i = 0; i < b.numel; ++i)
    {
        EXPECT_FLOAT_EQ(b.grad()[i], 1.0f);
    }
}

TEST(TensorOps, MulBackward)
{
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

TEST(TensorOps, ReluBackward)
{
    ParameterStore ps;
    ps.clear_tape();
    auto x = ps.tensor({4});
    fill_vec(x.data(), {-1.f, 0.2f, 3.f, -0.5f});
    auto y = relu(x, ps);
    auto s = sum(y, ps);
    ps.zero_grad();
    ps.backward(s);
    std::vector<float> expect = {0.f, 1.f, 1.f, 0.f};
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(x.grad()[i], expect[i]);
}

TEST(TensorOps, SumBackward)
{
    ParameterStore ps;
    ps.clear_tape();
    auto x = ps.tensor({3});
    fill_vec(x.data(), {1.f, 2.f, 3.f});
    auto s = sum(x, ps);
    ps.zero_grad();
    ps.backward(s);
    for (int i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(x.grad()[i], 1.f);
}

TEST(TensorOps, MatmulBackward)
{
    ParameterStore ps;
    ps.clear_tape();
    // A[2,2], B[2,2]
    auto A = ps.tensor({2, 2});
    auto B = ps.tensor({2, 2});
    fill_vec(A.data(), {1.f, 2.f, 3.f, 4.f}); // rows: [1,2], [3,4]
    fill_vec(B.data(), {5.f, 6.f, 7.f, 8.f}); // rows: [5,6], [7,8]
    auto C = matmul(A, B, ps);
    auto s = sum(C, ps);
    ps.zero_grad();
    ps.backward(s);
    // dA[m,k] = sum_n B[k,n] => row sums of B rows
    float r0 = B.data()[0] + B.data()[1]; // 5+6=11
    float r1 = B.data()[2] + B.data()[3]; // 7+8=15
    EXPECT_FLOAT_EQ(A.grad()[0], r0);
    EXPECT_FLOAT_EQ(A.grad()[1], r1);
    EXPECT_FLOAT_EQ(A.grad()[2], r0);
    EXPECT_FLOAT_EQ(A.grad()[3], r1);
    // dB[k,n] = sum_m A[m,k] => column sums of A columns
    float c0 = A.data()[0] + A.data()[2]; // 1+3=4
    float c1 = A.data()[1] + A.data()[3]; // 2+4=6
    // row 0 of B grad all = c0; row 1 all = c1
    EXPECT_FLOAT_EQ(B.grad()[0], c0);
    EXPECT_FLOAT_EQ(B.grad()[1], c0);
    EXPECT_FLOAT_EQ(B.grad()[2], c1);
    EXPECT_FLOAT_EQ(B.grad()[3], c1);
}

TEST(TensorOps, AddRowwiseBackward)
{
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
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(X.grad()[i], 1.f);
    // grad wrt b: [N, N]
    EXPECT_FLOAT_EQ(b.grad()[0], 3.f);
    EXPECT_FLOAT_EQ(b.grad()[1], 3.f);
}

TEST(TensorOps, SigmoidGradMatchesAnalytic)
{
    ParameterStore ps;
    ps.clear_tape();
    auto x = ps.tensor({3});
    fill_vec(x.data(), {-1.f, 0.0f, 2.f});
    auto y = sigmoid(x, ps);
    auto s = sum(y, ps);
    ps.zero_grad();
    ps.backward(s);
    for (int i = 0; i < 3; ++i)
    {
        float sig = 1.0f / (1.0f + std::exp(-x.data()[i]));
        float expected = sig * (1.0f - sig);
        EXPECT_NEAR(x.grad()[i], expected, 1e-5);
    }
}

TEST(TensorOps, LogGrad)
{
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
