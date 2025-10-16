#include "data/swedish_auto.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

SwedishAutoInsurance::SwedishAutoInsurance(std::string filename)
    : filename(std::move(filename)) {
  std::ifstream file(this->filename);
  if (!file.is_open()) {
    std::cout << "Could not open file for reading: " << this->filename
              << std::endl;
    throw std::runtime_error("FILE_NOT_FOUND");
  }

  std::vector<std::pair<float, float>> parsedData;
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
  auto train_size = static_cast<size_t>(total * 0.8);
  data.train_data.reserve(train_size);
  data.train_labels.reserve(train_size);
  data.test_data.reserve(total - train_size);
  data.test_labels.reserve(total - train_size);
  for (size_t i = 0; i < total; ++i) {
    if (i < train_size) {
      data.train_data.push_back(parsedData[i].first);
      data.train_labels.push_back(parsedData[i].second);
    } else {
      data.test_data.push_back(parsedData[i].first);
      data.test_labels.push_back(parsedData[i].second);
    }
  }
}

void SwedishAutoInsurance::summary() {
  std::cout << "Train data size: " << data.train_data.size() << std::endl;
  std::cout << "Train labels size: " << data.train_labels.size() << std::endl;
  std::cout << "Test data size: " << data.test_data.size() << std::endl;
  std::cout << "Test labels size: " << data.test_labels.size() << std::endl;
}
