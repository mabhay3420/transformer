#include "data/text.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

std::string load_text_data(std::string filename) {
  std::ifstream file(filename, std::ios::ate);
  if (!file) throw std::runtime_error("open failed");
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string data(size, '\0');
  file.read(&data[0], size);
  return data;
}

