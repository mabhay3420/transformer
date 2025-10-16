#pragma once

#include <string>
#include <utility>
#include <vector>

typedef std::vector<float> MNIST_IN;
typedef float MNIST_OUT;
typedef std::vector<MNIST_IN> MNIST_INS;
typedef std::vector<MNIST_OUT> MNIST_OUTS;
typedef std::pair<MNIST_INS, MNIST_OUTS> MNIST_BATCH;

struct MnistDataset {
  std::vector<std::vector<float>> train_data;
  std::vector<float> train_labels;
  std::vector<std::vector<float>> test_data;
  std::vector<float> test_labels;
};

struct MNIST {
  MnistDataset data;

  MNIST(int max_lines = 100, std::string train_csv = "",
        std::string test_csv = "");

  void summary();

 private:
  std::string train_csv;
  std::string test_csv;
  MNIST_BATCH load_data(const std::string& filename, int max_lines);
};
