#include "vis.hpp"
#include "mempool.hpp"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>

void build(const MemPoolIndex value, graph &g) {
  auto [nodes, edges, mem_pool] = g;
  if (nodes.find(value) != nodes.end())
    return;
  // if (g.0.find(value) != g.first.end())
  //   return;
  nodes.insert(value);
  auto v_value = mem_pool->get(value);
  for (auto child : v_value->children) {
    edges.insert({value, child});
    build(child, g);
  }
}

graph trace(const MemPoolIndex root, std::shared_ptr<MemPool<Value>> mem_pool) {
  snode nodes;
  sedge edges;
  graph g = {nodes, edges, mem_pool};
  build(root, g);
  return g;
}

void to_dot(const graph &g, std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }
  auto [nodes, edges, mem_pool] = g;
  file << "digraph G {\n";
  file << "concentrate = true;\n";
  file << "rankdir = LR;\n";
  for (const auto &node_i : nodes) {
    auto node = mem_pool->get(node_i);
    auto addr = reinterpret_cast<uintptr_t>(node);
    std::string recordInfo = "shape = record";
    if (node->persistent)
      recordInfo += ", color = red";
    recordInfo += "];\n";
    file << "  \"" << addr << "\" [label=\"" << node->label << "| data "
         << std::setprecision(3) << node->data << " | grad " << node->grad
         << "\"" << recordInfo;
    if (node->op != "" && node->op != " ")
      file << "  \"" << addr << node->op << "\" [label=\"" << node->op
           << "\"];\n";
  }
  for (const auto &edge : edges) {
    auto node_first = mem_pool->get(edge.first);
    auto node_second = mem_pool->get(edge.second);
    auto first_addr = reinterpret_cast<uintptr_t>(node_first);
    auto second_addr = reinterpret_cast<uintptr_t>(node_second);
    // op to parent node
    file << "\"" << first_addr << node_first->op << "\" -> \"" << first_addr
         << "\";\n";

    // child to op
    file << "\"" << second_addr << "\" -> \"" << first_addr << node_first->op
         << "\";\n";
  }
  file << "}\n";
  file.close();
}