#include "utils.hpp"
#include <fstream>
#include <iostream>

void dumpJson(json &j, std::string filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cout << "Could not open file for writing: " << filename << std::endl;
    throw std::runtime_error("FILE_NOT_FOUND");
  }
  file << j.dump(4) << std::endl;
  file.close();
}
void dumpJson(json &j, const char *filename) {
  std::string str(filename);
  dumpJson(j, str);
}

void dumpValues(std::vector<V> values, std::string filename) {
  json j = values;
  dumpJson(j, filename);
}
void dumpValues(std::vector<V> values, const char *filename) {
  std::string str(filename);
  dumpValues(values, str);
}
void dumpValues(std::vector<std::pair<V, V>> values, std::string filename) {
  json j = values;
  dumpJson(j, filename);
}
void dumpValues(std::vector<std::pair<V, V>> values, const char *filename) {
  std::string str(filename);
  dumpValues(values, str);
}