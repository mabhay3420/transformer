#include "dataloader.hpp"

#include <_stdlib.h>

#include <__config>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "mlx/data/core/CSVReader.h"

std::string load_text_data(std::string filename) {
  std::ifstream file(filename, std::ios::ate);
  if (!file) throw std::runtime_error("open failed");
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string data(size, '\0');
  file.read(&data[0], size);
  return data;
}

void Sampler::sample(Batch &batch, bool is_train) {
  auto get_random_index = [](int size) {
    unsigned seed = 1234;
    return rand_r(&seed) % size;  // using rand_r for thread safety
  };
  auto get_random_block = [&](const std::vector<int> &data) {
    int start_index = get_random_index(data.size() - block_size);
    return std::vector<int>(data.begin() + start_index,
                            data.begin() + start_index + block_size);
  };
  auto data = is_train ? train_data : val_data;
  // for each batch, sample a random block of data
  for (uint32_t i = 0; i < batch_size; ++i) {
    unsigned seed = batch_size + i;  // unique seed for each batch
    auto rnd_idx = rand_r(&seed) % (data.size() - block_size);
    // get a block of data
    auto train_seq = std::vector<int>(data.begin() + rnd_idx,
                                      data.begin() + rnd_idx + block_size);
    auto target = std::vector<int>(data.begin() + rnd_idx + 1,
                                   data.begin() + rnd_idx + block_size + 1);
    batch.first.push_back(train_seq);
    batch.second.push_back(target);
  }
}

MNIST::MNIST(int max_lines, std::string train_csv, std::string test_csv)
    : train_csv(std::move(train_csv)), test_csv(std::move(test_csv)) {
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

MNIST_BATCH MNIST::load_data(const std::string &csv_filename, int max_lines) {
  MNIST_INS images;
  MNIST_OUTS labels;

  std::string csv_contents = load_text_data(csv_filename);
  if (csv_contents.empty()) {
    return {std::move(images), std::move(labels)};
  }

  size_t total_rows =
      static_cast<size_t>(std::count(csv_contents.begin(), csv_contents.end(), '\n'));
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

  auto csv_stream = std::make_shared<std::istringstream>(std::move(csv_contents));
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

SwedishAutoInsurance::SwedishAutoInsurance(std::string filename)
    : filename(filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "Could not open file for reading: " << filename << std::endl;
    throw std::runtime_error("FILE_NOT_FOUND");
  }

  std::vector<std::pair<float, float>> parsedData;
  // read line by line
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(ss, token, ',')) {
      tokens.push_back(token);
    }
    parsedData.push_back({std::stof(tokens[0]), std::stof(tokens[1])});
  }
  auto total = parsedData.size();
  auto train_size = total * 0.8;
  data.train_data.reserve(train_size);
  data.train_labels.reserve(train_size);
  data.test_data.reserve(total - train_size);
  data.test_labels.reserve(total - train_size);
  for (int i = 0; i < total; i++) {
    if (i < train_size) {
      data.train_data.push_back(parsedData[i].first);
      data.train_labels.push_back(parsedData[i].second);
    } else {
      data.test_data.push_back(parsedData[i].first);
      data.test_labels.push_back(parsedData[i].second);
    }
  }
};

void SwedishAutoInsurance::summary() {
  std::cout << "Train data size: " << data.train_data.size() << std::endl;
  std::cout << "Train labels size: " << data.train_labels.size() << std::endl;
  std::cout << "Test data size: " << data.test_data.size() << std::endl;
  std::cout << "Test labels size: " << data.test_labels.size() << std::endl;
}
