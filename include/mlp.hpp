#ifndef MLP_HPP
#define MLP_HPP

#include "neuron.hpp"
#include "nlohmann/json.hpp"
#include <memory>
#include <ostream>
#include <string_view>

using nlohmann::json;
struct Layer {
  std::vector<std::shared_ptr<Neuron>> neurons;
  Layer(int in_dim, int out_dim);

  std::vector<V> operator()(const std::vector<V> &x);
  std::vector<V> params();
  friend std::ostream &operator<<(std::ostream &os, const Layer &);
  friend std::ostream &operator<<(std::ostream &os,
                                  const std::shared_ptr<Layer>);
  void dump(const int indent = 4);
};
void to_json(json &j, const Layer &l);
void to_json(json &j, const std::weak_ptr<Layer> l);

struct MLP {
  int in_dim;
  std::vector<int> out_dim;
  std::vector<std::shared_ptr<Layer>> layers;
  MLP(int in_dim, std::vector<int> out_dim);
  std::vector<V> operator()(const std::vector<V> &x);
  std::vector<V> params();
  friend std::ostream &operator<<(std::ostream &os, const MLP &);
  void dump(const int indent = 4);
  void dump(std::string filename, const int indent = 4);
};

void to_json(json &j, const MLP &m);

#endif // MLP_HPP