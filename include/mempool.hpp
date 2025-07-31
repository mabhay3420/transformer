#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
struct Value;

typedef size_t MemPoolIndex;

struct MemPoolEntry {
  size_t id;
  bool persistent = false;
};
template <typename T> struct MemPool {
  size_t persitent_boundary = 0;
  std::vector<T> mem;
  MemPool(MemPoolIndex size) : mem(size) { mem.reserve(size); }

  MemPoolIndex alloc() {
    mem.push_back(T());
    mem.back().id = mem.size() - 1;
    return mem.size() - 1;
  }
  MemPoolIndex size() { return mem.size(); }
  T *get(size_t i) { return &mem[i]; }
  std::vector<T *> get(std::vector<size_t> i) {
    std::vector<T *> out;
    for (auto j : i) {
      out.push_back(&mem[j]);
    }
    return out;
  }
  void set_persistent_boundary() {
    std::cout << "Setting persitent boundary to: " << mem.size() << std::endl;
    persitent_boundary = mem.size();
  }

  void reset() {
    // resize mem to persitent_boundary
    // std::cout << "Resetting mempool from size: " << mem.size() << " -> "
    //           << persitent_boundary << std::endl;
    mem.resize(persitent_boundary);
  }
};