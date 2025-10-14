#pragma once

#include <string>

#include "nlohmann/json_fwd.hpp"

using json = nlohmann::json;

void dumpJson(json &j, const std::string &filename);
void dumpJson(json &j, const char *filename);

float get_random_float(float min, float max);

