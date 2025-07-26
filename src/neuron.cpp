#include "neuron.hpp"
#include "utils.hpp"
#include <nlohmann/json.hpp>
#include <ostream>

using nlohmann::json;
Neuron::Neuron(int dim, bool with_activation)
    : d(dim), with_activation(with_activation) {
  auto get_random = []() {
    auto rnd = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return 2 * rnd - 1;
  };
  b = val(get_random());
  b->is_param = true;
  w.resize(dim);
  // fill with random values between -1 and 1
  for (int i = 0; i < dim; i++) {
    w[i] = val(get_random());
    w[i]->is_param = true;
  }
}

V Neuron::operator()(const std::vector<V> &x) {
  auto sum = b;
  for (int i = 0; i < d; i++) {
    sum = sum + w[i] * x[i];
  }
  if (with_activation) {
    auto act = sum->tanh();
    return act;
  }
  return sum;
}

std::vector<V> Neuron::params() {
  auto params = w;
  params.push_back(b);
  return params;
}

std::ostream &operator<<(std::ostream &os, const Neuron &n) {
  os << "Neuron(d=" << n.d << ", b=" << n.b << ", w=" << n.w << ")";
  return os;
}
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Neuron> n) {
  os << *n;
  return os;
}
void to_json(json &j, const Neuron &n) {
  j = json{{"dim", n.d}, {"b", n.b}, {"w", n.w}};
}
void to_json(json &j, const std::shared_ptr<Neuron> n) { j = *n; }