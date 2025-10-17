#include "utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace {

class EnvVarGuard {
 public:
  EnvVarGuard(const char* name, const char* value) : name_(name) {
    const char* existing = std::getenv(name);
    if (existing) {
      had_old_ = true;
      old_value_ = existing;
    }
    if (value) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }

  EnvVarGuard(const EnvVarGuard&) = delete;
  EnvVarGuard& operator=(const EnvVarGuard&) = delete;

  ~EnvVarGuard() {
    if (had_old_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool had_old_ = false;
};

}  // namespace

TEST(UtilsEnvironment, GetenvIntParsesValue) {
  EnvVarGuard guard("UTILS_TEST_INT", "42");
  EXPECT_EQ(getenv_int("UTILS_TEST_INT", 7), 42);
}

TEST(UtilsEnvironment, GetenvIntFallbackOnMissing) {
  EnvVarGuard guard("UTILS_TEST_INT_MISSING", nullptr);
  EXPECT_EQ(getenv_int("UTILS_TEST_INT_MISSING", 13), 13);
}

TEST(UtilsEnvironment, GetenvIntIgnoresInvalid) {
  EnvVarGuard guard("UTILS_TEST_INT_INVALID", "not-a-number");
  EXPECT_EQ(getenv_int("UTILS_TEST_INT_INVALID", -5), -5);
}

TEST(UtilsEnvironment, GetenvFloatParsesValue) {
  EnvVarGuard guard("UTILS_TEST_FLOAT", "3.5");
  EXPECT_FLOAT_EQ(getenv_float("UTILS_TEST_FLOAT", 0.25f), 3.5f);
}

TEST(UtilsEnvironment, GetenvFloatFallbackOnMissing) {
  EnvVarGuard guard("UTILS_TEST_FLOAT_MISSING", nullptr);
  EXPECT_FLOAT_EQ(getenv_float("UTILS_TEST_FLOAT_MISSING", 1.25f), 1.25f);
}

TEST(UtilsEnvironment, GetenvFloatIgnoresInvalid) {
  EnvVarGuard guard("UTILS_TEST_FLOAT_INVALID", "abc");
  EXPECT_FLOAT_EQ(getenv_float("UTILS_TEST_FLOAT_INVALID", -0.75f), -0.75f);
}

TEST(UtilsTensorHelpers, FillOneHotSetsSingleEntry) {
  ParameterStore store;
  auto tensor = store.tensor({2, 3}, TensorInit::ZeroData);
  fill_one_hot(tensor, 1, 2);

  const float* data = tensor.data();
  ASSERT_NE(data, nullptr);
  EXPECT_FLOAT_EQ(data[1 * 3 + 2], 1.0f);
  for (int i = 0; i < tensor.shape[0]; ++i) {
    for (int j = 0; j < tensor.shape[1]; ++j) {
      if (!(i == 1 && j == 2)) {
        EXPECT_FLOAT_EQ(data[i * tensor.shape[1] + j], 0.0f);
      }
    }
  }
}

TEST(UtilsTensorHelpers, FillOneHotClearsExistingRowValues) {
  ParameterStore store;
  auto tensor = store.tensor({2, 3}, TensorInit::ZeroData);
  float* data = tensor.data();
  ASSERT_NE(data, nullptr);
  for (size_t idx = 0; idx < tensor.numel; ++idx) {
    data[idx] = 0.5f;
  }

  fill_one_hot(tensor, 0, 1);

  EXPECT_FLOAT_EQ(data[0], 0.0f);
  EXPECT_FLOAT_EQ(data[1], 1.0f);
  EXPECT_FLOAT_EQ(data[2], 0.0f);
  EXPECT_FLOAT_EQ(data[3], 0.5f);
  EXPECT_FLOAT_EQ(data[4], 0.5f);
  EXPECT_FLOAT_EQ(data[5], 0.5f);
}

TEST(UtilsTensorHelpers, FillOneHotIgnoresInvalidIndices) {
  ParameterStore store;
  auto tensor = store.tensor({2, 3}, TensorInit::ZeroData);
  const float* before = tensor.data();
  fill_one_hot(tensor, 5, 1);   // invalid row
  fill_one_hot(tensor, 1, -1);  // invalid column
  const float* after = tensor.data();

  ASSERT_EQ(before, after);
  for (size_t idx = 0; idx < tensor.numel; ++idx) {
    EXPECT_FLOAT_EQ(after[idx], 0.0f);
  }
}

TEST(UtilsLogits, ArgmaxFindsLargestIndex) {
  const float logits[] = {-2.0f, 3.5f, 3.499f, 1.0f};
  EXPECT_EQ(argmax_from_logits(logits, 4), 1);
}

TEST(UtilsLogits, ArgmaxReturnsZeroWhenEmpty) {
  EXPECT_EQ(argmax_from_logits(nullptr, 0), 0);
}

TEST(UtilsLogits, SoftmaxNormalizesProbabilities) {
  const float logits[] = {0.0f, 0.0f, 0.0f};
  auto probs = softmax_from_logits(logits, 3);
  ASSERT_EQ(probs.size(), 3u);

  float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
  const float expected = 1.0f / 3.0f;
  for (float p : probs) {
    EXPECT_NEAR(p, expected, 1e-5f);
  }
}

TEST(UtilsLogits, SoftmaxHandlesLargeValues) {
  const float logits[] = {1000.0f, 0.0f};
  auto probs = softmax_from_logits(logits, 2);
  ASSERT_EQ(probs.size(), 2u);

  EXPECT_NEAR(probs[0], 1.0f, 1e-6f);
  EXPECT_NEAR(probs[1], 0.0f, 1e-6f);
}

TEST(UtilsRandom, RandomFloatWithinRange) {
  srand(12345);
  const float min = -2.0f;
  const float max = 5.0f;
  for (int i = 0; i < 10; ++i) {
    float value = get_random_float(min, max);
    EXPECT_GE(value, min);
    EXPECT_LE(value, max);
  }
}
