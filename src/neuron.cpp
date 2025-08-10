#include "neuron.hpp"
#include "mempool.hpp"
#include "utils.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>
#include <random>

using nlohmann::json;
Neuron::Neuron(int dim, std::shared_ptr<MemPool<Value>> mem_pool,
               bool with_activation, bool with_bias, Activation act)
    : d(dim), with_activation(with_activation), with_bias(with_bias),
      mem_pool(mem_pool), act(act) {
  w.resize(dim);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, dim + 1);
  auto get_random = [&generator, &distribution, dim]() {
    auto rnd = distribution(generator);
    auto rnd_f = static_cast<float>(rnd) / (dim + 1);
    return 2 * rnd_f - 1;
  };
  if (with_bias) {
    // fill with random values between -1 and 1
    b = val(get_random(), mem_pool);
    mem_pool->get(b)->is_param = true;
  }
  for (int i = 0; i < dim; i++) {
    w[i] = val(get_random(), mem_pool);
    mem_pool->get(w[i])->is_param = true;
  }
}

MemPoolIndex Neuron::operator()(const std::vector<MemPoolIndex> &x) {
  auto sum = val(0.0f, mem_pool);
  if (with_bias) {
    auto sum = b;
  }
  for (int i = 0; i < d; i++) {
    auto y = mul(w[i], x[i], mem_pool);
    sum = add(sum, y, mem_pool);
  }
  if (with_activation) {
    MemPoolIndex actResult;
    switch (act) {
    case Activation::RELU:
      actResult = relu(sum, mem_pool);
      return actResult;
    case Activation::TANH:
      actResult = tanh(sum, mem_pool);
      return actResult;
    default:
      return sum; // No activation
    }
    auto act = relu(sum, mem_pool);
    return act;
  }
  return sum;
}
std::vector<MemPoolIndex>
Neuron::operator()(const std::vector<std::vector<MemPoolIndex>> &x) {
  std::vector<MemPoolIndex> out;
  // TODO - Can be parallelized
  for (const auto &xi : x) {
    auto y = this->operator()(xi);
    out.push_back(y);
  }
  return out;
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