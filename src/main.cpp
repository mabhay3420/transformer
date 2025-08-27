#include "bigramnn.hpp"
#include "embednlp.hpp"
#include "linear_regression.hpp"
#include "utils.hpp"
#include "xormodel.hpp"
#include "xormodel_tensors.hpp"
#include <chrono>
#include <fstream>
int main() {
  auto start = std::chrono::high_resolution_clock::now();
#ifdef DEBUG
  auto total = 1;
#else
  auto total = 1;
#endif
  // LinearRegression();
  for (int i = 0; i < total; i++) {
    // XORLinearRegression();
    XORWithTensors();
  }
  // MnistDnn();
  // BigraLm();
  // BigramNN();
  // EmbedNLP();
  auto end = std::chrono::high_resolution_clock::now();
  using time_unit = std::chrono::duration<double, std::milli>;
  auto duration = std::chrono::duration_cast<time_unit>(end - start);
  std::ofstream fout("time.txt", std::ios::app);
  fout << duration.count() / total << std::endl;
  return 0;
}
