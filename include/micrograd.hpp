#ifndef MICROGRAD_HPP
#define MICROGRAD_HPP

#include "nlohmann/json_fwd.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>
#include <vector>

using nlohmann::json;
typedef std::function<void()> BackpropCallback;

struct Value;
using V = std::shared_ptr<Value>;
using VWP = std::weak_ptr<Value>;
struct Value : std::enable_shared_from_this<Value> {
  float data = 0.0f;
  float grad = 0.0f;
  int in_degree = 0;
  bool is_param = false;
  std::vector<VWP> children; // to keep track of dependencies
  std::string op;
  std::string label;
  // TODO - Use this later
  bool requires_grad = true;
  BackpropCallback backward = []() {};

  explicit Value(float data) : data(data){};
  Value(float data, std::vector<VWP> children, std::string op,
        std::string label = "")
      : data(data), children(children), op(op), label(label) {}
  Value(float data, std::vector<V> children, std::string op,
        std::string label = "")
      : data(data), op(op), label(label) {
    for (auto child : children) {
      this->children.push_back(VWP(child));
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Value &v);
  friend std::ostream &operator<<(std::ostream &os, V v);

  V self() { return shared_from_this(); }
  V add(const V &rhs) {
    auto out = std::make_shared<Value>(data + rhs->data,
                                       std::vector{self(), rhs}, "+");
    this->in_degree += 1;
    rhs->in_degree += 1;
    out->backward = [lhs = self(), rhs, out]() {
      lhs->grad += 1.0f * out->grad;
      rhs->grad += 1.0f * out->grad;
    };
    return out;
  }
  V mul(const V &rhs) {
    auto out = std::make_shared<Value>(data * rhs->data,
                                       std::vector{self(), rhs}, "*");
    this->in_degree += 1;
    rhs->in_degree += 1;
    out->backward = [lhs = self(), rhs, out]() {
      lhs->grad += rhs->data * out->grad;
      rhs->grad += lhs->data * out->grad;
    };
    return out;
  }

  V sub(const V &rhs) {
    auto out = std::make_shared<Value>(data - rhs->data,
                                       std::vector{self(), rhs}, "-");
    this->in_degree += 1;
    rhs->in_degree += 1;
    out->backward = [lhs = self(), rhs, out]() {
      lhs->grad += 1.0f * out->grad;
      rhs->grad -= 1.0f * out->grad;
    };
    return out;
  }

  V div(const V &rhs) {
    if (rhs->data == 0.0f) {
      throw std::runtime_error("Division by zero");
    }
    auto out = std::make_shared<Value>(data / rhs->data,
                                       std::vector{self(), rhs}, "/");
    this->in_degree += 1;
    rhs->in_degree += 1;
    out->backward = [lhs = self(), rhs, out]() {
      lhs->grad += out->grad / rhs->data;
      rhs->grad += (-lhs->data * out->grad / (rhs->data * rhs->data));
    };
    return out;
  }

  V tanh() {
    auto out =
        std::make_shared<Value>(std::tanh(data), std::vector{self()}, "tanh");
    this->in_degree += 1;
    // dtanh(x) = 1 - tanh(x)^2
    out->backward = [lhs = self(), out]() {
      lhs->grad += out->grad * (1.0f - out->data * out->data);
    };
    return out;
  }

  V exp() {
    auto out =
        std::make_shared<Value>(std::exp(data), std::vector{self()}, "exp");
    this->in_degree += 1;
    out->backward = [lhs = self(), out]() {
      lhs->grad += out->grad * out->data;
    };
    return out;
  }

  V log() {
    if (this->data <= 0.0f) {
      throw std::runtime_error("Taking Log of a non-positive number");
    }
    auto out =
        std::make_shared<Value>(std::log(data), std::vector{self()}, "log");
    this->in_degree += 1;
    out->backward = [lhs = self(), out]() {
      if (lhs->data == 0.0f) {
        throw std::runtime_error("Division by zero");
      }
      lhs->grad += out->grad / lhs->data;
    };
    return out;
  }
};

V max(const std::vector<V> &x);
std::vector<V> softmax(const std::vector<V> &x);
std::vector<V> one_hot_encode(int category, int total_categories);

inline V val(float x) { return std::make_shared<Value>(x); }
inline V val(float x, std::vector<VWP> children, std::string op,
             std::string label = "") {
  return std::make_shared<Value>(x, std::move(children), op, label);
}
inline V operator+(const V &a, const V &b) { return a->add(b); }
inline V operator-(const V &a, const V &b) { return a->sub(b); }
inline V operator*(const V &a, const V &b) { return a->mul(b); }
inline V operator/(const V &a, const V &b) { return a->div(b); }
inline V tanh(const V &a) { return a->tanh(); }
inline V exp(const V &a) { return a->exp(); }
inline V log(const V &a) { return a->log(); }

// floating point versions
// template version with any integer type
inline V operator+(const V &a, float x) { return a->add(val(x)); }
inline V operator-(const V &a, float x) { return a->sub(val(x)); }
inline V operator*(const V &a, float x) { return a->mul(val(x)); }
inline V operator/(const V &a, float x) { return a->div(val(x)); }

// and reversed
inline V operator+(float x, const V &a) { return a->add(val(x)); }
inline V operator-(float x, const V &a) { return a->sub(val(x)); }
inline V operator*(float x, const V &a) { return a->mul(val(x)); }
inline V operator/(float x, const V &a) { return a->div(val(x)); }

void backprop(const V root);
void to_json(json &j, const V &v);

#endif
