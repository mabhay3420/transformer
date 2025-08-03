#include "neuron.hpp"
#include "mempool.hpp"
#include "utils.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>

using nlohmann::json;
Neuron::Neuron(int dim, std::shared_ptr<MemPool<Value>> mem_pool,
               bool with_activation)
    : d(dim), with_activation(with_activation), mem_pool(mem_pool) {
  auto get_random = []() {
    auto rnd = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return 2 * rnd - 1;
  };
  b = val(get_random(), mem_pool);
  w.resize(dim);
  mem_pool->get(b)->is_param = true;
  // fill with random values between -1 and 1
  for (int i = 0; i < dim; i++) {
    w[i] = val(get_random(), mem_pool);
    mem_pool->get(w[i])->is_param = true;
  }
}

size_t Neuron::operator()(const std::vector<MemPoolIndex> &x) {
  auto sum = b;
  for (int i = 0; i < d; i++) {
    auto y = mul(w[i], x[i], mem_pool);
    sum = add(sum, y, mem_pool);
  }
  if (with_activation) {
    auto act = tanh(sum, mem_pool);
    return act;
  }
  return sum;
}

std::vector<MemPoolIndex> Neuron::params() {
  auto params = w;
  params.push_back(b);
  return params;
}

std::ostream &operator<<(std::ostream &os, const Neuron &n) {
  os << "Neuron(d=" << n.d << ", b=" << n.mem_pool->get(n.b)
     << ", w=" << n.mem_pool->get(n.w) << ")";
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