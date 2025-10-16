#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"
#include "tensor.hpp"

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

int getenv_int(const char *name, int fallback) {
  if (!name) return fallback;
  if (const char *value = std::getenv(name)) {
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end != value) return static_cast<int>(parsed);
  }
  return fallback;
}

float getenv_float(const char *name, float fallback) {
  if (!name) return fallback;
  if (const char *value = std::getenv(name)) {
    char *end = nullptr;
    float parsed = std::strtof(value, &end);
    if (end != value) return parsed;
  }
  return fallback;
}

std::string getenv_str(const char *name, const std::string &fallback) {
  if (!name) return fallback;
  if (const char *value = std::getenv(name)) {
    if (value[0] != '\0') return std::string(value);
  }
  return fallback;
}

void fill_one_hot(Tensor &tensor, int row, int index) {
  if (tensor.shape.size() != 2) return;
  if (row < 0 || row >= tensor.shape[0]) return;
  if (index < 0 || index >= tensor.shape[1]) return;
  float *ptr = tensor.data();
  int stride = tensor.shape[1];
  ptr[row * stride + index] = 1.0f;
}

int argmax_from_logits(const float *logits, int size) {
  if (!logits || size <= 0) return 0;
  int best_idx = 0;
  float best_val = logits[0];
  for (int i = 1; i < size; ++i) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  return best_idx;
}

std::vector<float> softmax_from_logits(const float *logits, int size) {
  std::vector<float> probs(size);
  if (!logits || size <= 0) return probs;
  float max_logit = logits[0];
  for (int i = 1; i < size; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    float val = std::exp(logits[i] - max_logit);
    probs[i] = val;
    sum += val;
  }
  if (sum <= 0.0f) {
    float inv = 1.0f / std::max(1, size);
    for (int i = 0; i < size; ++i) {
      probs[i] = inv;
    }
    return probs;
  }
  for (int i = 0; i < size; ++i) {
    probs[i] /= sum;
  }
  return probs;
}
