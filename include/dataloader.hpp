#pragma once
#include <cassert>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace std;
template <typename T>
void split_data(const float ratio, const std::vector<T> &data,
                std::vector<T> &train_data, std::vector<T> &val_data) {
  assert(ratio > 0.0f && ratio < 1.0f && "Ratio must be between 0 and 1");
  auto split_index = static_cast<size_t>(data.size() * ratio);
  train_data.assign(data.begin(), data.begin() + split_index);
  val_data.assign(data.begin() + split_index, data.end());
  if (train_data.empty() || val_data.empty()) {
    throw std::runtime_error("Split resulted in empty dataset");
  }
}

string load_text_data(string filename);

/*
Batch 1:
[ [1, 2, 3, 4, 5, 6, 7, 8], 1 ]

Batch 2:
[ [9, 10, 11, 12, 13, 14, 15, 16], 2 ]
 ...

*/
typedef std::vector<std::vector<int>> vvint;
typedef std::vector<std::vector<float>> vvfloat;
typedef std::pair<vvint, vvint> Batch;  // (data, target)
struct Sampler {
  uint32_t batch_size;  // to be processed in parallel
  uint32_t block_size;  // context length

  std::vector<int> train_data;
  std::vector<int> val_data;

  //   TODO - Is this copying things?
  //   TODO - If yes then figure out something else
  Sampler(size_t batch_size, size_t block_size,
          const std::vector<int> &train_data, const std::vector<int> &val_data)
      : batch_size(batch_size),
        block_size(block_size),
        train_data(train_data),
        val_data(val_data) {}

  void sample(Batch &batch, bool is_train = true);
};

typedef vector<float> MNIST_IN;
typedef float MNIST_OUT;
typedef vector<MNIST_IN> MNIST_INS;
;
typedef vector<MNIST_OUT> MNIST_OUTS;
typedef std::pair<MNIST_INS, MNIST_OUTS> MNIST_BATCH;

struct MnistDataset {
  vector<vector<float>> train_data;
  vector<float> train_labels;
  vector<vector<float>> test_data;
  vector<float> test_labels;
};
struct MNIST {
  MnistDataset data;

  MNIST(int max_lines = 100, std::string train_csv = "data/mnist_train.csv",
        std::string test_csv = "data/mnist_test.csv");

  void summary();

 private:
  std::string train_csv;
  std::string test_csv;
  MNIST_BATCH load_data(std::string filename, int max_lines);
};

struct SwedishAutoInsuranceData {
  std::vector<float> train_data;
  std::vector<float> train_labels;
  std::vector<float> test_data;
  std::vector<float> test_labels;
};

struct SwedishAutoInsurance {
  SwedishAutoInsuranceData data;
  SwedishAutoInsurance(std::string filename = "data/swedish_auto_insurace.csv");
  void summary();

 private:
  std::string filename;
};

template <typename InputDataType, typename LableType,
          typename InputMemPoolIndexType, typename LabelMemPoolIndexType>
std::vector<std::pair<InputMemPoolIndexType, LabelMemPoolIndexType>>
getRandomBatch(
    const std::vector<InputDataType> &x, const std::vector<LableType> &y,
    std::function<InputMemPoolIndexType(const InputDataType &)> inputTransform,
    std::function<LabelMemPoolIndexType(const LableType &)> labelTransform,
    int batch_size, int start_index = 0, int end_index = -1) {
  if (end_index == -1) end_index = x.size();
  std::vector<int> indices(batch_size);
  auto total_size = x.size();
  for (int i = 0; i < batch_size; i++) {
    indices[i] = (rand() % (end_index - start_index)) + start_index;
  }
  std::vector<std::pair<InputMemPoolIndexType, LabelMemPoolIndexType>> batch;
  for (auto i : indices) {
    batch.push_back({inputTransform(x[i]), labelTransform(y[i])});
  }
  return batch;
}
