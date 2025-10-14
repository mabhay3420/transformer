#include "utils.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"

void dumpJson(json &j, const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cout << "Could not open file for writing: " << filename << std::endl;
    throw std::runtime_error("FILE_NOT_FOUND");
  }
  file << j.dump(4) << std::endl;
}

void dumpJson(json &j, const char *filename) {
  dumpJson(j, std::string(filename));
}

float get_random_float(float min, float max) {
  return static_cast<float>(rand()) / RAND_MAX * (max - min) + min;
}
