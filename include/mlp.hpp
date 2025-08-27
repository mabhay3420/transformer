#ifndef MLP_HPP
#define MLP_HPP

#include <memory>
#include <ostream>
#include <string_view>

#include "mempool.hpp"
#include "neuron.hpp"
#include "nlohmann/json.hpp"

using nlohmann::json;
struct Layer {
  MemPool<Value> *mem_pool;
  std::vector<Neuron *> neurons;
  Layer(int in_dim, int out_dim, MemPool<Value> *mem_pool,
        bool with_activation = true, bool with_bias = true,
        Activation act = Activation::RELU);

  std::vector<MemPoolIndex> operator()(const std::vector<MemPoolIndex> &x);
  std::vector<std::vector<MemPoolIndex>> operator()(
      const std::vector<std::vector<MemPoolIndex>> &x);
  std::vector<size_t> params();
  friend std::ostream &operator<<(std::ostream &os, const Layer &);
  friend std::ostream &operator<<(std::ostream &os, const Layer *);
  void dump(const int indent = 4);
};
void to_json(json &j, const Layer &l);
void to_json(json &j, const std::weak_ptr<Layer> l);

struct MLP {
  int in_dim;
  MemPool<Value> *mem_pool;
  std::vector<int> out_dim;
  std::vector<Layer *> layers;
  MLP(int in_dim, std::vector<int> out_dim, MemPool<Value> *mem_pool,
      bool last_with_activation = true, bool with_bias = true,
      Activation act = Activation::RELU);
  std::vector<size_t> operator()(const std::vector<size_t> &x);
  std::vector<size_t> params();
  friend std::ostream &operator<<(std::ostream &os, const MLP &);
  void dump(const int indent = 4);
  void dump(std::string filename, const int indent = 4);
};

void to_json(json &j, const MLP &m);

#endif  // MLP_HPP