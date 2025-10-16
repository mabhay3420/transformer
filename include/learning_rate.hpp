#pragma once
#include <cmath>

template <typename T>
struct LRScheduler {
  float init_lr;
  float curr_lr;
  int cnt;

  LRScheduler(float init_lr) : init_lr(init_lr), curr_lr(init_lr), cnt(0) {};

  float get() { return static_cast<T*>(this)->get(); }
  float getLog() { return static_cast<T*>(this)->getLog(); }
};

struct ConstantLRScheduler : LRScheduler<ConstantLRScheduler> {
  ConstantLRScheduler(float init_lr)
      : LRScheduler<ConstantLRScheduler>(init_lr) {};
  float get() { return init_lr; }
  float getLog() { return init_lr; }
};

struct ExpLinspaceLRScheduler : LRScheduler<ExpLinspaceLRScheduler> {
  float base;
  float start;
  float end;
  float limit;
  float steps;
  using LRScheduler<ExpLinspaceLRScheduler>::cnt;
  using LRScheduler<ExpLinspaceLRScheduler>::curr_lr;
  ExpLinspaceLRScheduler(float start, float end, float steps,
                         float limit = -4.0f, float base = 10.0f)
      : LRScheduler<ExpLinspaceLRScheduler>(std::pow(base, start)),
        start(start),
        end(end),
        steps(steps),
        limit(limit),
        base(base) {}
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

struct StepLRScheduler : LRScheduler<StepLRScheduler> {
  int cliff;
  float gamma;
  float limit;

  StepLRScheduler(float init_lr, int cliff, float gamma, float limit = 1e-4f)
      : LRScheduler<StepLRScheduler>(init_lr),
        cliff(cliff),
        gamma(gamma),
        limit(limit) {}

  float get();
  float getLog() { return curr_lr; }
};