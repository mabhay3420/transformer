#include "mlp.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <ostream>

Layer::Layer(int in_dim, int out_dim) {
  for (int i = 0; i < out_dim; i++) {
    neurons.push_back(std::make_shared<Neuron>(in_dim));
  }
}

std::ostream &operator<<(std::ostream &os, const Layer &l) {
  os << "Layer(neurons=[";
  for (const auto &n : l.neurons) {
    os << n << ",";
  }
  os << "])\n";
  return os;
}
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Layer> l) {
  os << *l;
  return os;
}

std::vector<V> Layer::operator()(const std::vector<V> &x) {
  std::vector<V> result;
  for (int i = 0; i < neurons.size(); i++) {
    auto n = *neurons[i];
    result.push_back(n(x));
  }
  return result;
}

std::vector<V> Layer::params() {
  std::vector<V> params;
  for (auto n : neurons) {
    auto p = n->params();
    params.insert(params.end(), p.begin(), p.end());
  }
  return params;
}

MLP::MLP(int in_dim, std::vector<int> out_dim)
    : in_dim(in_dim), out_dim(out_dim) {
  std::vector<int> layers_dim = {in_dim};
  layers_dim.insert(layers_dim.end(), out_dim.begin(), out_dim.end());
  for (int i = 0; i < layers_dim.size() - 1; i++) {
    layers.push_back(std::make_shared<Layer>(layers_dim[i], layers_dim[i + 1]));
  };
}

std::vector<V> MLP::operator()(const std::vector<V> &x) {
  std::vector<V> result(x);
  for (int i = 0; i < layers.size(); i++) {
    auto l = *layers[i];
    result = l(result);
  }
  return result;
}
std::vector<V> MLP::params() {
  std::vector<V> params;
  for (auto l : layers) {
    auto p = l->params();
    params.insert(params.end(), p.begin(), p.end());
  }
  return params;
}

std::ostream &operator<<(std::ostream &os, const MLP &m) {
  Indent indent;
  os << indent << "MLP(\n";
  os << "MLP(\n\tin_dim=" << m.in_dim << ", \n\tout_dim=" << m.out_dim
     << ",\n\tlayers=[\n";
  for (const auto &l : m.layers) {
    os << l << ",\n";
  }
  os << "])";
  return os;
}

void to_json(json &j, const Layer &l) {
  j = json{
      {"num_neurons", l.neurons.size()},
      {"neurons", l.neurons},
  };
}
void to_json(json &j, const std::shared_ptr<Layer> l) { j = *l; }
void to_json(json &j, const MLP &mlp) {
  j = json{
      {"in_dim", mlp.in_dim},
      {"out_dim", mlp.out_dim},
      {"num_layers", mlp.layers.size()},
      {"layers", mlp.layers},
  };
}

void Layer::dump(const int indent) {
  json j = *this;
  std::cout << j.dump(indent) << std::endl;
  std::cout << std::endl;
}
void MLP::dump(const int indent) {
  json j = *this;
  std::cout << j.dump(indent) << std::endl;
  std::cout << std::endl;
}

void MLP::dump(std::string filename, const int indent) {
  json j = *this;
  //   std::ofstream file(filename);
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }
  file << j.dump(indent) << std::endl;
  file.close();
}