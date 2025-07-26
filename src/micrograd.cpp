#include "micrograd.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
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

V max(const std::vector<V> &xs) {
  V max = xs[0];
  for (auto x : xs) {
    if (x->data > max->data) {
      max = x;
    }
  }
  return max;
}

std::vector<V> softmax(const std::vector<V> &x) {
  // NOTE: not doing the max version for now
  auto xmax = max(x);
  std::vector<V> exps;
  for (auto xi : x) {
    auto xnorm = xi - xmax;
    exps.push_back(exp(xnorm));
  }
  auto sum = val(0.0f);
  for (auto exp : exps) {
    sum = sum + exp;
  }
  for (int i = 0; i < exps.size(); i++) {
    exps[i] = exps[i] / sum;
  }
  return exps;
}
std::vector<V> one_hot_encode(int category, int total_categories) {
  std::vector<V> out;
  for (int i = 0; i < total_categories; i++) {
    if (i == category) {
      out.push_back(val(1.0f));
    } else {
      out.push_back(val(0.0f));
    }
  }
  return out;
}

void build_topo(const V root, std::vector<V> &topo) {
  // invalid node
  if (!root)
    return;
  std::stack<V> next;
  std::set<V> visited;
  next.push(root);
  while (!next.empty()) {
    bool all_children_visited = true;
    for (auto child : next.top()->children) {
      if (visited.find(child.lock()) != visited.end()) {
        continue;
      }
      visited.insert(child.lock());
      next.push(child.lock());
      all_children_visited = false;
    }
    if (all_children_visited) {
      topo.push_back(next.top());
      next.pop();
    }
  }
}

void build_topo_kahn(const V root, std::vector<V> &topo) {
  std::set<V> all;
  std::set<V> S;
  std::unordered_map<V, int> in_degree;
  // by definition this is the only node that has no incoming edge
  // do a bfs and collect nodes with in_degree = 0
  S.insert(root);
  // invariant: All that are inserted in S have in_degree = 0
  // i.e. there are no parents left to visit
  while (!S.empty()) {
    auto v = S.begin();
    auto N = *v;
    S.erase(v);
    topo.push_back(N);
    for (auto M : N->children) {
      in_degree[M.lock()] += 1;
      if (in_degree[M.lock()] == M.lock()->in_degree) {
        S.insert(M.lock());
      }
    }
  }
}

void backprop(const V root) {
  root->grad = 1.0f;
  std::vector<V> topo;
  // build_topo_kahn(root, topo);
  build_topo(root, topo);
  std::reverse(topo.begin(), topo.end());
  // std::cout << "Total Nodes: " << topo.size() << std::endl;
  for (auto v : topo) {
    v->backward();
  }
}

void to_json(json &j, const V &v) { j = v->data; }