#ifndef NEURON_HPP
#define NEURON_HPP

#include "micrograd.hpp"
#include <_stdlib.h>
#include <memory>
#include <vector>

struct Neuron {
  int d;            // dimension
  std::vector<V> w; // weight
  V b;              // bias
  bool with_activation;
  Neuron(int dim, bool with_activation = true);

  V operator()(const std::vector<V> &x);

  std::vector<V> params();

  friend std::ostream &operator<<(std::ostream &os, const Neuron &);
  friend std::ostream &operator<<(std::ostream &os,
                                  const std::shared_ptr<Neuron>);
};
void to_json(json &j, const Neuron &n);
void to_json(json &j, const std::shared_ptr<Neuron> n);
#endif
