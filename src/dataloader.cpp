#include <__config>
#include <_stdlib.h>
#include <cassert>
#include <fstream>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "dataloader.hpp"

std::string load_text_data(std::string filename) {
  std::ifstream file(filename, std::ios::ate);
  if (!file)
    throw std::runtime_error("open failed");
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string data(size, '\0');
  file.read(&data[0], size);
  return data;
}


void Sampler::sample(Batch &batch, bool is_train) {

  auto get_random_index = [](int size) {
    unsigned seed = 1234;
    return rand_r(&seed) % size; // using rand_r for thread safety
  };
  auto get_random_block = [&](const std::vector<int> &data) {
    int start_index = get_random_index(data.size() - block_size);
    return std::vector<int>(data.begin() + start_index,
                            data.begin() + start_index + block_size);
  };
  auto data = is_train ? train_data : val_data;
  // for each batch, sample a random block of data
  for (uint32_t i = 0; i < batch_size; ++i) {
    unsigned seed = batch_size + i; // unique seed for each batch
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

MNIST_BATCH MNIST::load_data(std::string filename, int max_lines) {
  MNIST_INS ins;
  MNIST_OUTS labels;
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading");
  }
  // values are comma separated
  std::string line;
  int num_lines = 0;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(ss, token, ',')) {
      tokens.push_back(token);
    }
    // train_data.push_back(std::stof(tokens[0]));
    labels.push_back(std::stoi(tokens[0]));
    MNIST_IN in;
    for (int i = 1; i < tokens.size(); i++) {
      in.push_back(std::stof(tokens[i]) / 255);
    }
    ins.push_back(in);
    num_lines++;
    if (num_lines > max_lines)
      break;
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