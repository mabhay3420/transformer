#ifndef MICROGRAD_HPP
#define MICROGRAD_HPP

#include "mempool.hpp"
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
struct Value {
  float data = 0.0f;
  float grad = 0.0f;
  std::vector<MemPoolIndex> children; // to keep track of dependencies
  std::string op;
  std::string label;
  bool is_param = false;
  // TODO - Use this later
  // bool requires_grad = true;
  BackpropCallback backward = []() {};

  Value() = default;
  ~Value() = default;
  explicit Value(float data) : data(data){};
  Value(float data, std::vector<MemPoolIndex> children, std::string op,
        std::string label = "")
      : data(data), children(children), op(op), label(label){};
  friend std::ostream &operator<<(std::ostream &os, const Value &v);
  friend std::ostream &operator<<(std::ostream &os, const Value *v);
  friend std::ostream &operator<<(std::ostream &os,
                                  const std::vector<Value *> &v);
  friend std::ostream &operator<<(std::ostream &os, V v);
};

V max(const std::vector<V> &x);
std::vector<MemPoolIndex> softmax(const std::vector<MemPoolIndex> &x,
                                  std::shared_ptr<MemPool<Value>> mem_pool);
std::vector<MemPoolIndex> one_hot_encode(int category, int total_categories);
size_t argmax(const std::vector<MemPoolIndex> &xs,
              std::shared_ptr<MemPool<Value>> mem_pool);

inline MemPoolIndex val(float x, std::shared_ptr<MemPool<Value>> mem_pool) {
  auto v = mem_pool->alloc();
  mem_pool->get(v)->data = x;
  return v;
}
inline MemPoolIndex val(float x, std::vector<MemPoolIndex> children,
                        std::string op,
                        std::shared_ptr<MemPool<Value>> mem_pool,
                        std::string label = "") {
  auto v = mem_pool->alloc();
  auto p = mem_pool->get(v);
  p->data = x;
  p->children = children;
  p->op = op;
  p->label = label;
  return v;
}
MemPoolIndex add(const MemPoolIndex &a, const MemPoolIndex &b,
                 std::shared_ptr<MemPool<Value>> mem_pool);
MemPoolIndex sub(const MemPoolIndex &a, const MemPoolIndex &b,
                 std::shared_ptr<MemPool<Value>> mem_pool);
MemPoolIndex mul(const MemPoolIndex &a, const MemPoolIndex &b,
                 std::shared_ptr<MemPool<Value>> mem_pool);
MemPoolIndex div(const MemPoolIndex &a, const MemPoolIndex &b,
                 std::shared_ptr<MemPool<Value>> mem_pool);
MemPoolIndex tanh(const MemPoolIndex &a,
                  std::shared_ptr<MemPool<Value>> mem_pool);
MemPoolIndex exp(const MemPoolIndex &a,
                 std::shared_ptr<MemPool<Value>> mem_pool);
MemPoolIndex log(const MemPoolIndex &a,
                 std::shared_ptr<MemPool<Value>> mem_pool);

void backprop(const MemPoolIndex root,
              std::shared_ptr<MemPool<Value>> mem_pool);
void to_json(json &j, const V &v);
void to_json(json &j, const Value *v);

#endif
