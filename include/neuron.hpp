#ifndef NEURON_HPP
#define NEURON_HPP

#include <_stdlib.h>

#include <memory>
#include <vector>

#include "mempool.hpp"
#include "micrograd.hpp"

enum class Activation { RELU, TANH };

struct Neuron {
  int d;                        // dimension
  std::vector<MemPoolIndex> w;  // weight
  MemPoolIndex b;               // bias
  bool with_activation;
  bool with_bias;
  Activation act;
  MemPool<Value> *mem_pool;
  Neuron(int dim, MemPool<Value> *mem_pool, bool is_activation = true,
         bool with_bias = true, Activation act = Activation::RELU);

  MemPoolIndex operator()(const std::vector<MemPoolIndex> &x);
  std::vector<MemPoolIndex> operator()(
      const std::vector<std::vector<MemPoolIndex>> &x);

  std::vector<MemPoolIndex> params();

  friend std::ostream &operator<<(std::ostream &os, const Neuron &);
  friend std::ostream &operator<<(std::ostream &os, const Neuron *);
};
void to_json(json &j, const Neuron &n);
void to_json(json &j, const Neuron *n);
#endif
