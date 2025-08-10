#include "bigram.hpp"
#include "dataloader.hpp"
#include "linear_regression.hpp"
#include "micrograd.hpp"
#include "mlp.hpp"
#include "mnist.hpp"
#include "neuron.hpp"
#include "probs.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include "vis.hpp"
#include "xormodel.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>

#define LABEL(I) I->label = #I

float f(float x) { return 3 * x * x - 4 * x + 5; }
float df(float x) {
  float h = 0.00001f;
  return (f(x + h) - f(x - h)) / (2 * h);
}
float dfreal(float x) {
  //   return 6 * x - 4;
  return 6 * x - 4;
}

int main() {
  // LinearRegression();
  // XORLinearRegression();
  // MnistDnn();
  BigraLm();
  return 0;
}