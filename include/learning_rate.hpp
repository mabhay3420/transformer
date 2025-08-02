#pragma once

template <typename T> struct LRScheduler {
  float init_lr;
  float curr_lr;
  int cnt;

  LRScheduler(float init_lr) : init_lr(init_lr), curr_lr(init_lr), cnt(0){};

  float get() { return static_cast<T *>(this)->get(); }
};

struct StepLRScheduler : LRScheduler<StepLRScheduler> {
  int cliff;
  float gamma;
  float limit;

  StepLRScheduler(float init_lr, int cliff, float gamma, float limit = 1e-4f)
      : LRScheduler<StepLRScheduler>(init_lr), cliff(cliff), gamma(gamma),
        limit(limit) {}

  float get();
};