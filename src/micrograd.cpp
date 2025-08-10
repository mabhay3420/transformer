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

size_t argmax(const std::vector<MemPoolIndex> &xs,
              std::shared_ptr<MemPool<Value>> mem_pool) {
  auto max_idx = 0;
  for (int i = 1; i < xs.size(); i++) {
    auto xi = mem_pool->get(xs[i]);
    if (xi->data > mem_pool->get(xs[max_idx])->data) {
      max_idx = i;
    }
  }
  return max_idx;
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
  auto xmax = max(x, mem_pool);
  auto xmax_v = val(mem_pool->get(xmax)->data, mem_pool);
  std::vector<MemPoolIndex> exps;
  for (auto xi : x) {
    auto xnorm = sub(xi, xmax_v, mem_pool);
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
  mem_pool->get(o_i)->data =
      mem_pool->get(a_i)->data + mem_pool->get(b_i)->data;
  mem_pool->get(o_i)->children = {a_i, b_i};
  mem_pool->get(o_i)->op = "+";
  mem_pool->get(o_i)->backward = [a_i, b_i, o_i, m = mem_pool.get()]() {
    m->get(a_i)->grad += 1.0f * m->get(o_i)->grad;
    m->get(b_i)->grad += 1.0f * m->get(o_i)->grad;
  };
  return o_i;
}

MemPoolIndex sub(const MemPoolIndex &a_i, const MemPoolIndex &b_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  // auto a = mem_pool->get(a_i);
  // auto b = mem_pool->get(b_i);
  // auto o = mem_pool->get(o_i);
  mem_pool->get(o_i)->data =
      mem_pool->get(a_i)->data - mem_pool->get(b_i)->data;
  mem_pool->get(o_i)->children = {a_i, b_i};
  mem_pool->get(o_i)->op = "-";
  mem_pool->get(o_i)->backward = [a_i, b_i, o_i, m = mem_pool.get()]() {
    m->get(a_i)->grad += 1.0f * m->get(o_i)->grad;
    m->get(b_i)->grad -= 1.0f * m->get(o_i)->grad;
  };
  return o_i;
}

MemPoolIndex mul(const MemPoolIndex &a_i, const MemPoolIndex &b_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  mem_pool->get(o_i)->data =
      mem_pool->get(a_i)->data * mem_pool->get(b_i)->data;
  mem_pool->get(o_i)->children = {a_i, b_i};
  mem_pool->get(o_i)->op = "*";
  mem_pool->get(o_i)->backward = [a_i, b_i, o_i, m = mem_pool.get()]() {
    m->get(a_i)->grad += m->get(b_i)->data * m->get(o_i)->grad;
    m->get(b_i)->grad += m->get(a_i)->data * m->get(o_i)->grad;
  };
  return o_i;
}

MemPoolIndex div(const MemPoolIndex &a_i, const MemPoolIndex &b_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto epsilon = 1e-8f;
  auto o_i = mem_pool->alloc();
  mem_pool->get(o_i)->data =
      mem_pool->get(a_i)->data / (mem_pool->get(b_i)->data + epsilon);
  mem_pool->get(o_i)->children = {a_i, b_i};
  mem_pool->get(o_i)->op = "/";
  mem_pool->get(o_i)->backward = [a_i, b_i, o_i, m = mem_pool.get(),
                                  epsilon]() {
    m->get(a_i)->grad += m->get(o_i)->grad / (m->get(b_i)->data + epsilon);
    m->get(b_i)->grad -= (m->get(a_i)->data * m->get(o_i)->grad) /
                         (m->get(b_i)->data * m->get(b_i)->data + epsilon);
  };
  return o_i;
}

MemPoolIndex tanh(const MemPoolIndex &a_i,
                  std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  mem_pool->get(o_i)->data = std::tanh(mem_pool->get(a_i)->data);
  mem_pool->get(o_i)->children = {a_i};
  mem_pool->get(o_i)->backward = [a_i, o_i, m = mem_pool.get()]() {
    m->get(a_i)->grad +=
        m->get(o_i)->grad * (1.0f - m->get(o_i)->data * m->get(o_i)->data);
  };
  return o_i;
}

MemPoolIndex relu(const MemPoolIndex &a_i,
                  std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  mem_pool->get(o_i)->data =
      mem_pool->get(a_i)->data > 0.0f ? mem_pool->get(a_i)->data : 0.0f;
  mem_pool->get(o_i)->children = {a_i};
  mem_pool->get(o_i)->backward = [a_i, o_i, m = mem_pool.get()]() {
    m->get(a_i)->grad +=
        m->get(o_i)->grad * (m->get(o_i)->data > 0.0f ? 1.0f : 0.0f);
  };
  return o_i;
}

MemPoolIndex exp(const MemPoolIndex &a_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  auto o_i = mem_pool->alloc();
  mem_pool->get(o_i)->data = std::exp(mem_pool->get(a_i)->data);
  mem_pool->get(o_i)->children = {a_i};
  mem_pool->get(o_i)->backward = [a_i, o_i, m = mem_pool.get()]() {
    m->get(a_i)->grad += m->get(o_i)->grad * m->get(o_i)->data;
  };
  return o_i;
}

MemPoolIndex log(const MemPoolIndex &a_i,
                 std::shared_ptr<MemPool<Value>> mem_pool) {
  float epsilon = 1e-8f;
  auto o_i = mem_pool->alloc();
  mem_pool->get(o_i)->data = std::log(mem_pool->get(a_i)->data + epsilon);
  mem_pool->get(o_i)->children = {a_i};
  mem_pool->get(o_i)->backward = [a_i, o_i, m = mem_pool.get(), epsilon]() {
    m->get(a_i)->grad += m->get(o_i)->grad / (m->get(a_i)->data + epsilon);
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
  // update params now
  for (auto iter = mem_pool->persistent.rbegin();
       iter != mem_pool->persistent.rend(); ++iter) {
    iter->backward();
  }
}

void to_json(json &j, const V &v) { j = v->data; }
void to_json(json &j, const Value *v) { j = v->data; }