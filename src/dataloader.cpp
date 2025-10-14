#include "dataloader.hpp"

#include <_stdlib.h>

#include <__config>
#include <cassert>
#include <fstream>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>

#include "mlx/data/Array.h"
#include "mlx/data/Sample.h"
#include "mlx/data/stream/CSVReader.h"

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
    : train_csv(train_csv), test_csv(test_csv) {
  auto train_data = load_data(train_csv, max_lines);
  auto test_data = load_data(test_csv, max_lines);
  data = {
      .train_data = train_data.first,
      .train_labels = train_data.second,
      .test_data = test_data.first,
      .test_labels = test_data.second,
  };
};
void MNIST::summary() {
  std::cout << "Train data size: " << data.train_data.size() << std::endl;
  std::cout << "Train labels size: " << data.train_labels.size() << std::endl;
  std::cout << "Test data size: " << data.test_data.size() << std::endl;
  std::cout << "Test labels size: " << data.test_labels.size() << std::endl;
}

MNIST_BATCH MNIST::load_data(std::string csv_filename, int max_lines) {
  MNIST_INS ins;
  MNIST_OUTS labels;
  std::ifstream file(csv_filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading");
  }

  std::string csv_body((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  if (csv_body.empty()) {
    return {ins, labels};
  }

  auto newline_pos = csv_body.find('\n');
  std::string first_line =
      (newline_pos == std::string::npos) ? csv_body : csv_body.substr(0, newline_pos);
  if (first_line.empty()) {
    return {ins, labels};
  }

  int num_columns =
      static_cast<int>(std::count(first_line.begin(), first_line.end(), ',')) + 1;
  if (num_columns < 2) {
    throw std::runtime_error("MNIST CSV is expected to have at least two columns");
  }

  std::ostringstream header;
  header << "label";
  std::vector<std::string> pixel_keys;
  pixel_keys.reserve(num_columns - 1);
  for (int i = 1; i < num_columns; ++i) {
    std::string key = "pixel" + std::to_string(i - 1);
    header << ',' << key;
    pixel_keys.push_back(std::move(key));
  }
  header << '\n';

  std::string csv_with_header = header.str();
  csv_with_header += csv_body;
  auto csv_stream = std::make_shared<std::istringstream>(std::move(csv_with_header));
  mlx::data::stream::CSVReader reader(csv_stream);

  int num_lines = 0;
  while (max_lines <= 0 || num_lines < max_lines) {
    auto sample = reader.next();
    if (sample.empty()) break;

    auto label_array =
        mlx::data::sample::check_key(sample, "label", mlx::data::ArrayType::Int8);
    std::string label_str(
        reinterpret_cast<char*>(label_array->data()), label_array->size());
    labels.push_back(static_cast<float>(std::stoi(label_str)));

    MNIST_IN pixels;
    pixels.reserve(pixel_keys.size());
    for (const auto& key : pixel_keys) {
      auto value_array =
          mlx::data::sample::check_key(sample, key, mlx::data::ArrayType::Int8);
      std::string value_str(
          reinterpret_cast<char*>(value_array->data()), value_array->size());
      pixels.push_back(std::stof(value_str) / 255.0f);
    }
    ins.push_back(std::move(pixels));
    ++num_lines;
  }

  return {ins, labels};
}

SwedishAutoInsurance::SwedishAutoInsurance(std::string filename)
    : filename(filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "Could not open file for reading: " << filename << std::endl;
    throw std::runtime_error("FILE_NOT_FOUND");
  }

  std::vector<pair<float, float>> parsedData;
  // read line by line
  string line;
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
