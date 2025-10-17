#include "learning_rate.hpp"

#include <algorithm>
#include <iostream>
float StepLRScheduler::get() {
  cnt++;
  if (cnt % cliff == 0 && curr_lr > limit) {
    std::cout << "Reducing learning rate by factor of " << gamma << std::endl;
    float next = curr_lr * gamma;
    curr_lr = std::max(limit, next);
    if (curr_lr != next) {
      std::cout << "Clamped learning rate to limit: " << curr_lr << std::endl;
    } else {
      std::cout << "New learning rate: " << curr_lr << std::endl;
    }
  }
  return curr_lr;
}
