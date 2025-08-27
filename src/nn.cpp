#include "nn.hpp"

#include <algorithm>

namespace nn {

Linear::Linear(int in_f, int out_f, ParameterStore &store, bool bias,
               float init_scale, unsigned seed)
    : in_features(in_f),
      out_features(out_f),
      use_bias(bias),
      W(store.parameter({in_f, out_f}, init_scale, seed)),
      b(bias ? store.parameter({out_f}, init_scale, seed ^ 0xA5A5A5)
             : Tensor{}) {}

Tensor Linear::forward(const Tensor &x, ParameterStore &store) {
  auto y = matmul(x, W, store);
  if (use_bias) {
    y = add_rowwise(y, b, store);
  }
  return y;
}

std::vector<Tensor> Linear::params() {
  if (use_bias) return {W, b};
  return {W};
}

Tensor Tanh::forward(const Tensor &x, ParameterStore &store) {
  return vtanh(x, store);
}

Tensor Sigmoid::forward(const Tensor &x, ParameterStore &store) {
  return sigmoid(x, store);
}

Tensor Sequential::forward(const Tensor &x, ParameterStore &store) {
  Tensor h = x;
  for (auto &m : layers) {
    h = m->forward(h, store);
  }
  return h;
}

std::vector<Tensor> Sequential::params() {
  std::vector<Tensor> all;
  for (auto &m : layers) {
    auto p = m->params();
    all.insert(all.end(), p.begin(), p.end());
  }
  return all;
}

Tensor bce_with_logits_loss(const Tensor &logits, const Tensor &targets,
                            ParameterStore &store, float eps) {
  auto probs = sigmoid(logits, store);
  Tensor ones = store.tensor(targets.shape);
  std::fill(ones.data(), ones.data() + ones.numel, 1.0f);
  Tensor epsT = store.tensor(targets.shape);
  std::fill(epsT.data(), epsT.data() + epsT.numel, eps);
  auto p_eps = add(probs, epsT, store);
  auto q = sub(ones, probs, store);
  auto q_eps = add(q, epsT, store);
  auto term1 = mul(targets, vlog(p_eps, store), store);
  auto one_minus_y = sub(ones, targets, store);
  auto term2 = mul(one_minus_y, vlog(q_eps, store), store);
  auto sum_terms = add(term1, term2, store);
  auto s = sum(sum_terms, store);
  Tensor scale = store.tensor({1});
  scale.data()[0] = -1.0f / static_cast<float>(targets.shape[0]);
  return mul(s, scale, store);
}

}  // namespace nn
