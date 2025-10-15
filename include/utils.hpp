#pragma once

#include <string>
#include <vector>

#include "nlohmann/json_fwd.hpp"

using json = nlohmann::json;

struct Tensor;

void dumpJson(json &j, const std::string &filename);
void dumpJson(json &j, const char *filename);

float get_random_float(float min, float max);

int getenv_int(const char *name, int fallback);
float getenv_float(const char *name, float fallback);

void fill_one_hot(Tensor &tensor, int row, int index);
int argmax_from_logits(const float *logits, int size);
std::vector<float> softmax_from_logits(const float *logits, int size);
