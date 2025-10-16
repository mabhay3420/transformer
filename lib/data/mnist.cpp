#include "data/mnist.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "data/text.hpp"
#include "mlx/data/core/CSVReader.h"
#include "utils.hpp"

MNIST::MNIST(int max_lines, std::string train_csv, std::string test_csv)
    : train_csv(std::move(train_csv)), test_csv(std::move(test_csv)) {
  const std::string data_dir = getenv_str("MNIST_DATA_DIR", "data_tmp");
  if (this->train_csv.empty()) {
    this->train_csv = data_dir + "/mnist_train.csv";
  }
  if (this->test_csv.empty()) {
    this->test_csv = data_dir + "/mnist_test.csv";
  }
  auto train_batch = load_data(this->train_csv, max_lines);
  auto test_batch = load_data(this->test_csv, max_lines);
  data.train_data = std::move(train_batch.first);
  data.train_labels = std::move(train_batch.second);
  data.test_data = std::move(test_batch.first);
  data.test_labels = std::move(test_batch.second);
}

void MNIST::summary() {
  std::cout << "Train data size: " << data.train_data.size() << std::endl;
  std::cout << "Train labels size: " << data.train_labels.size() << std::endl;
  std::cout << "Test data size: " << data.test_data.size() << std::endl;
  std::cout << "Test labels size: " << data.test_labels.size() << std::endl;
}

MNIST_BATCH MNIST::load_data(const std::string& csv_filename, int max_lines) {
  MNIST_INS images;
  MNIST_OUTS labels;

  std::string csv_contents = load_text_data(csv_filename);
  if (csv_contents.empty()) {
    return {std::move(images), std::move(labels)};
  }

  size_t total_rows = static_cast<size_t>(
      std::count(csv_contents.begin(), csv_contents.end(), '\n'));
  if (!csv_contents.empty() && csv_contents.back() != '\n') {
    total_rows += 1;
  }
  if (total_rows == 0) {
    return {std::move(images), std::move(labels)};
  }
  if (max_lines > 0) {
    total_rows = std::min<size_t>(total_rows, static_cast<size_t>(max_lines));
  }
  images.reserve(total_rows);
  labels.reserve(total_rows);

  auto csv_stream =
      std::make_shared<std::istringstream>(std::move(csv_contents));
  mlx::data::core::CSVReader reader(csv_stream, ',', '"');

  size_t loaded = 0;
  while (loaded < total_rows) {
    auto row = reader.next();
    if (row.empty()) break;
    if (row.size() < 2) {
      throw std::runtime_error("MNIST CSV row must contain label and pixels");
    }

    labels.push_back(
        static_cast<float>(std::strtol(row.front().c_str(), nullptr, 10)));

    MNIST_IN pixels(row.size() - 1);
    for (size_t i = 1; i < row.size(); ++i) {
      pixels[i - 1] = std::strtof(row[i].c_str(), nullptr) / 255.0f;
    }
    images.push_back(std::move(pixels));
    ++loaded;
  }

  return {std::move(images), std::move(labels)};
}
