#ifndef NEURON_HPP
#define NEURON_HPP

#include "mempool.hpp"
#include "micrograd.hpp"
#include <_stdlib.h>
#include <memory>
#include <vector>

enum class Activation { RELU, TANH };

struct Neuron {
  int d;                       // dimension
  std::vector<MemPoolIndex> w; // weight
  MemPoolIndex b;              // bias
  bool with_activation;
  bool with_bias;
  Activation act;
  std::shared_ptr<MemPool<Value>> mem_pool;
  Neuron(int dim, std::shared_ptr<MemPool<Value>> mem_pool,
         bool is_activation = true, bool with_bias = true,
         Activation act = Activation::RELU);

  MemPoolIndex operator()(const std::vector<MemPoolIndex> &x);
  std::vector<MemPoolIndex>
  operator()(const std::vector<std::vector<MemPoolIndex>> &x);

  std::vector<MemPoolIndex> params();

  friend std::ostream &operator<<(std::ostream &os, const Neuron &);
  friend std::ostream &operator<<(std::ostream &os,
                                  const std::shared_ptr<Neuron>);
};
void to_json(json &j, const Neuron &n);
void to_json(json &j, const std::shared_ptr<Neuron> n);
#endif
