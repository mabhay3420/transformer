#include "micrograd.hpp"
#include "mempool.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <queue>
#include <set>
#include <unordered_map>

std::ostream &operator<<(std::ostream &os, const Value &v) {
  os << "Value(data=" << v.data << ")";
  return os;
}
std::ostream &operator<<(std::ostream &os, V v) {
  os << "Value(data=" << v->data << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Value *v) {
  os << "Value(data=" << v->data << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<Value *> &v) {
  os << "[";
  for (auto x : v) {
    os << x << ",";
  }
  os << "]";
  return os;
}

MemPoolIndex max(const std::vector<MemPoolIndex> &xs,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto max_i = xs[0];
  for (auto x_i : xs) {
    auto x = mem_pool->get(x_i);
    auto max = mem_pool->get(max_i);
    if (x->data > max->data) {
      max_i = x_i;
    }
  }
  return max_i;
}

std::vector<MemPoolIndex> softmax(const std::vector<MemPoolIndex> &x,
                                  std::shared_ptr<MemPool<Value>> mem_pool) {
  // NOTE: not doing the max version for now
  auto o_i = mem_pool->alloc();
  auto o = mem_pool->get(o_i);
  auto xmax = max(x, mem_pool);
  std::vector<MemPoolIndex> exps;
  for (auto xi : x) {
    auto xnorm = sub(xi, xmax, mem_pool);
    exps.push_back(exp(xnorm, mem_pool));
  }
  auto sum = val(0.0f, mem_pool);
  for (auto exp : exps) {
    sum = add(sum, exp, mem_pool);
  }
  for (auto &exp : exps) {
    exp = div(exp, sum, mem_pool);
  }
  return exps;
}

std::vector<MemPoolIndex>
one_hot_encode(int category, int total_categories,
               std::shared_ptr<MemPool<Value>> mem_pool) {
  std::vector<MemPoolIndex> out;
  for (int i = 0; i < total_categories; i++) {
    if (i == category) {
      out.push_back(val(1.0f, mem_pool));
    } else {
      out.push_back(val(0.0f, mem_pool));
    }
  }
  return out;
}

MemPoolIndex add(const MemPoolIndex &a_i, const MemPoolIndex &b_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  auto a = mem_pool->get(a_i);
  auto b = mem_pool->get(b_i);
  auto o = mem_pool->get(o_i);
  o->data = a->data + b->data;
  o->children = {a_i, b_i};
  o->op = "+";
  o->backward = [a_i, b_i, o_i, mem_pool]() {
    auto a = mem_pool->get(a_i);
    auto b = mem_pool->get(b_i);
    auto o = mem_pool->get(o_i);
    a->grad += 1.0f * o->grad;
    b->grad += 1.0f * o->grad;
  };
  return o_i;
}

MemPoolIndex sub(const MemPoolIndex &a_i, const MemPoolIndex &b_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  auto a = mem_pool->get(a_i);
  auto b = mem_pool->get(b_i);
  auto o = mem_pool->get(o_i);
  o->data = a->data - b->data;
  o->children = {a_i, b_i};
  o->op = "-";
  o->backward = [a_i, b_i, o_i, mem_pool]() {
    auto a = mem_pool->get(a_i);
    auto b = mem_pool->get(b_i);
    auto o = mem_pool->get(o_i);
    a->grad += 1.0f * o->grad;
    b->grad -= 1.0f * o->grad;
  };
  return o_i;
}

MemPoolIndex mul(const MemPoolIndex &a_i, const MemPoolIndex &b_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  auto a = mem_pool->get(a_i);
  auto b = mem_pool->get(b_i);
  auto o = mem_pool->get(o_i);
  o->data = a->data * b->data;
  o->children = {a_i, b_i};
  o->op = "*";
  o->backward = [a_i, b_i, o_i, mem_pool]() {
    auto a = mem_pool->get(a_i);
    auto b = mem_pool->get(b_i);
    auto o = mem_pool->get(o_i);
    a->grad += b->data * o->grad;
    b->grad += a->data * o->grad;
  };
  return o_i;
}

MemPoolIndex div(const MemPoolIndex &a_i, const MemPoolIndex &b_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  auto a = mem_pool->get(a_i);
  auto b = mem_pool->get(b_i);

  if (b->data == 0.0f) {
    throw std::runtime_error("Division by zero");
  }
  auto o = mem_pool->get(o_i);
  o->data = a->data / b->data;
  o->children = {a_i, b_i};
  o->op = "/";
  o->backward = [a_i, b_i, o_i, mem_pool]() {
    auto a = mem_pool->get(a_i);
    auto b = mem_pool->get(b_i);
    auto o = mem_pool->get(o_i);
    a->grad += o->data / b->data;
    b->grad -= a->data * o->data / (b->data * b->data);
  };
  return o_i;
}

MemPoolIndex tanh(const MemPoolIndex &a_i,
                  std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  auto a = mem_pool->get(a_i);
  auto b = mem_pool->get(o_i);
  b->data = std::tanh(a->data);
  b->children = {a_i};
  b->backward = [a_i, o_i, mem_pool]() {
    auto a = mem_pool->get(a_i);
    auto b = mem_pool->get(o_i);
    a->grad += b->data * (1.0f - b->data * b->data);
  };
  return o_i;
}

MemPoolIndex exp(const MemPoolIndex &a_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  auto a = mem_pool->get(a_i);
  auto b = mem_pool->get(o_i);
  b->data = std::exp(a->data);
  b->children = {a_i};
  b->backward = [a_i, o_i, mem_pool]() {
    auto a = mem_pool->get(a_i);
    auto b = mem_pool->get(o_i);
    a->grad += b->data * b->data;
  };
  return o_i;
}

MemPoolIndex log(const MemPoolIndex &a_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  auto a = mem_pool->get(a_i);
  auto b = mem_pool->get(o_i);
  b->data = std::log(a->data);
  b->children = {a_i};
  b->backward = [a_i, o_i, mem_pool]() {
    auto a = mem_pool->get(a_i);
    auto b = mem_pool->get(o_i);
    if (a->data == 0.0f) {
      throw std::runtime_error("Division by zero");
    }
    a->grad += b->data / a->data;
  };
  return o_i;
}

void backprop(const MemPoolIndex root,
              std::shared_ptr<MemPool<Value>> mem_pool) {
  mem_pool->get(root)->grad = 1.0f;
  // guaranteed to be in topological order
  for (auto iter = mem_pool->mem.rbegin(); iter != mem_pool->mem.rend();
       ++iter) {
    iter->backward();
  }
}

void to_json(json &j, const V &v) { j = v->data; }
void to_json(json &j, const Value *v) { j = v->data; }