#pragma once

#include <string>
#include <vector>

struct SwedishAutoInsuranceData {
  std::vector<float> train_data;
  std::vector<float> train_labels;
  std::vector<float> test_data;
  std::vector<float> test_labels;
};

struct SwedishAutoInsurance {
  SwedishAutoInsuranceData data;
  SwedishAutoInsurance(std::string filename = "data/swedish_auto_insurace.csv");
  void summary();

 private:
  std::string filename;
};
