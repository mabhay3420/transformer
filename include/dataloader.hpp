#ifndef DATALOADER_HPP
#define DATALOADER_HPP
#include <string>
#include <utility>
#include <vector>

using namespace std;

string load_text_data(string filename);

void split_data(const float ratio, const std::vector<int> &data,
                std::vector<int> &train_data, std::vector<int> &val_data);

/*
Batch 1:
[ [1, 2, 3, 4, 5, 6, 7, 8], 1 ]

Batch 2:
[ [9, 10, 11, 12, 13, 14, 15, 16], 2 ]
 ...

*/
typedef std::vector<std::vector<int>> vvint;
typedef std::vector<std::vector<float>> vvfloat;
typedef std::pair<vvint, vvint> Batch; // (data, target)
struct Sampler {

  uint32_t batch_size; // to be processed in parallel
  uint32_t block_size; // context length

  std::vector<int> train_data;
  std::vector<int> val_data;

  //   TODO - Is this copying things?
  //   TODO - If yes then figure out something else
  Sampler(size_t batch_size, size_t block_size,
          const std::vector<int> &train_data, const std::vector<int> &val_data)
      : batch_size(batch_size), block_size(block_size), train_data(train_data),
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

  MNIST(std::string train_csv = "data/mnist_train.csv",
        std::string test_csv = "data/mnist_test.csv", int max_lines = 100);

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
#endif