/**
 * @file learning_rate.hpp
 * @brief Learning rate schedulers for training.
 *
 * Provides various learning rate scheduling strategies including constant,
 * exponential, and step decay schedules.
 */

#pragma once
#include <cmath>
#include <stdexcept>

/**
 * @class LRScheduler
 * @brief Base class for learning rate schedulers.
 * @tparam T Derived scheduler type (CRTP pattern)
 */
template <typename T>
struct LRScheduler {
  float init_lr;  ///< Initial learning rate
  float curr_lr;  ///< Current learning rate
  int cnt;        ///< Step counter

  /**
   * @brief Construct scheduler with initial learning rate.
   * @param init_lr Initial learning rate value
   */
  LRScheduler(float init_lr) : init_lr(init_lr), curr_lr(init_lr), cnt(0) {};

  /**
   * @brief Get current learning rate.
   * @return Current learning rate
   */
  float get() { return static_cast<T*>(this)->get(); }

  /**
   * @brief Get current learning rate (alias for get).
   * @return Current learning rate
   */
  float getLog() { return static_cast<T*>(this)->getLog(); }
};

/**
 * @class ConstantLRScheduler
 * @brief Constant learning rate scheduler.
 *
 * Maintains the same learning rate throughout training.
 */
struct ConstantLRScheduler : LRScheduler<ConstantLRScheduler> {
  /**
   * @brief Construct constant scheduler.
   * @param init_lr Constant learning rate
   */
  ConstantLRScheduler(float init_lr)
      : LRScheduler<ConstantLRScheduler>(init_lr) {};

  /**
   * @brief Get learning rate (constant).
   * @return Constant learning rate
   */
  float get() { return init_lr; }

  /**
   * @brief Get learning rate (alias).
   * @return Constant learning rate
   */
  float getLog() { return init_lr; }
};

/**
 * @class ExpLinspaceLRScheduler
 * @brief Exponential linear space learning rate scheduler.
 *
 * Schedules learning rate as base^(start + t * step_size) where t increases
 * linearly over steps, providing exponential decay or growth.
 */
struct ExpLinspaceLRScheduler : LRScheduler<ExpLinspaceLRScheduler> {
  float base;   ///< Exponential base (default 10)
  float start;  ///< Starting exponent
  float end;    ///< Ending exponent
  float limit;  ///< Unused parameter
  float steps;  ///< Total number of steps
  using LRScheduler<ExpLinspaceLRScheduler>::cnt;
  using LRScheduler<ExpLinspaceLRScheduler>::curr_lr;

  /**
   * @brief Construct exponential scheduler.
   * @param start Starting exponent
   * @param end Ending exponent
   * @param steps Number of steps
   * @param limit Unused (default -4.0)
   * @param base Exponential base (default 10.0)
   */
  ExpLinspaceLRScheduler(float start, float end, float steps,
                         float limit = -4.0f, float base = 10.0f)
      : LRScheduler<ExpLinspaceLRScheduler>(std::pow(base, start)),
        start(start),
        end(end),
        steps(steps),
        limit(limit),
        base(base) {}

  /**
   * @brief Get current learning rate.
   * @return Exponential learning rate
   */
  float get() {
    cnt++;
    auto stepSize = std::abs(end - start) / steps;
    curr_lr = std::pow(base, start);
    if (cnt < steps) {
      if (start > end) {
        curr_lr = std::pow(base, start - (cnt * stepSize));
      } else {
        curr_lr = std::pow(base, start + (cnt * stepSize));
      }
    }
    return curr_lr;
  }

  float getLog() { return curr_lr; }
};

/**
 * @class StepLRScheduler
 * @brief Step decay learning rate scheduler.
 *
 * Reduces learning rate by gamma factor every cliff steps, with minimum limit.
 */
struct StepLRScheduler : LRScheduler<StepLRScheduler> {
  int cliff;    ///< Steps between decays
  float gamma;  ///< Decay factor
  float limit;  ///< Minimum learning rate

  /**
   * @brief Construct step scheduler.
   * @param init_lr Initial learning rate
   * @param cliff Steps between decays
   * @param gamma Decay factor (multiplied each cliff)
   * @param limit Minimum learning rate (default 1e-4)
   */
  StepLRScheduler(float init_lr, int cliff, float gamma, float limit = 1e-4f)
      : LRScheduler<StepLRScheduler>(init_lr),
        cliff(cliff),
        gamma(gamma),
        limit(limit) {
    if (cliff <= 0) {
      throw std::invalid_argument("StepLRScheduler cliff must be positive");
    }
    if (gamma <= 0.0f) {
      throw std::invalid_argument("StepLRScheduler gamma must be positive");
    }
  }

  /**
   * @brief Get current learning rate with step decay.
   * @return Decayed learning rate
   */
  float get();

  /**
   * @brief Get current learning rate (alias).
   * @return Current learning rate
   */
  float getLog() { return curr_lr; }
};
