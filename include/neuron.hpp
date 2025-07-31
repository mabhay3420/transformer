#ifndef NEURON_HPP
#define NEURON_HPP

#include "mempool.hpp"
#include "micrograd.hpp"
#include <_stdlib.h>
#include <memory>
#include <vector>

struct Neuron {
  int d;                       // dimension
  std::vector<MemPoolIndex> w; // weight
  size_t b;                    // bias
  bool with_activation;
  std::shared_ptr<MemPool<Value>> mem_pool;
  Neuron(int dim, std::shared_ptr<MemPool<Value>> mem_pool,
         bool is_activation = true);

  size_t operator()(const std::vector<size_t> &x);

  std::vector<MemPoolIndex> params();

  friend std::ostream &operator<<(std::ostream &os, const Neuron &);
  friend std::ostream &operator<<(std::ostream &os,
                                  const std::shared_ptr<Neuron>);
};
void to_json(json &j, const Neuron &n);
void to_json(json &j, const std::shared_ptr<Neuron> n);
#endif
