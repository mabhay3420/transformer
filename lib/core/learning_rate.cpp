#include "learning_rate.hpp"

#include <iostream>
float StepLRScheduler::get() {
  cnt++;
  if (cnt % cliff == 0 && curr_lr > limit) {
    std::cout << "Reducing learning rate by factor of " << gamma << std::endl;
    curr_lr *= gamma;
    std::cout << "New learning rate: " << curr_lr << std::endl;
  }
  return curr_lr;
}