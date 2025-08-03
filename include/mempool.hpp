#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
struct Value;

typedef size_t MemPoolIndex;

template <typename T> struct MemPool {
  bool persistent_done = false;
  std::vector<T> persistent;
  std::vector<T> mem;
  MemPool() = default;

  MemPoolIndex alloc() {
    if (persistent_done) {
      mem.push_back(T());
      return persistent.size() + mem.size() - 1;
    } else {
      persistent.push_back(T());
      return persistent.size() - 1;
    }
  }
  MemPoolIndex size() { return persistent.size() + mem.size(); }
  T *get(size_t i) {
    if (i < persistent.size())
      return &persistent[i];
    else
      return &mem[i - persistent.size()];
  }
  std::vector<T *> get(std::vector<size_t> i) {
    std::vector<T *> out;
    for (auto j : i) {
      out.push_back(this->get(j));
    }
    return out;
  }
  void set_persistent_boundary() { persistent_done = true; }
  void reset() { mem.clear(); }
  void clear() {
    persistent.clear();
    mem.clear();
  }
};