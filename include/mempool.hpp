#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
struct Value;

typedef size_t MemPoolIndex;

template <typename T> struct MemPool {
  bool persistent_done = false;
  std::vector<T *> persistent;
  std::vector<T *> mem;
  MemPool() = default;
  ~MemPool() { deallocate_all(); }

  MemPoolIndex alloc() {
    T *obj = new T();
    if (persistent_done) {
      mem.push_back(obj);
      return persistent.size() + mem.size() - 1;
    } else {
      persistent.push_back(obj);
      return persistent.size() - 1;
    }
  }
  MemPoolIndex size() { return persistent.size() + mem.size(); }
  T *get(size_t i) {
    if (i < persistent.size())
      return persistent[i];
    else
      return mem[i - persistent.size()];
  }
  std::vector<T *> get(const std::vector<size_t> &i) {
    std::vector<T *> out;
    for (auto j : i) {
      out.push_back(this->get(j));
    }
    return out;
  }
  void set_persistent_boundary() {
    if (persistent_done) {
      throw std::runtime_error("Persistent boundary already set");
    }
    persistent_done = true;
  }
  void deallocate_temp() {
    for (auto p : mem) {
      delete p;
    }
    mem.clear();
  }
  void deallocate_all() {
    for (auto p : persistent) {
      delete p;
    }
    for (auto p : mem) {
      delete p;
    }
    persistent.clear();
    mem.clear();
  }
};