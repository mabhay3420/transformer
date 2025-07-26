#include "vis.hpp"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>

void build(const V value, graph &g) {
  if (!value)
    return;
  if (g.first.find(value) != g.first.end())
    return;
  g.first.insert(value);
  for (auto child : value->children) {
    g.second.insert({value, child.lock()});
    build(child.lock(), g);
  }
}

graph trace(const V root) {
  snode nodes;
  sedge edges;
  graph g = {nodes, edges};
  build(root, g);
  return g;
}

void to_dot(const graph &g, std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }
  file << "digraph G {\n";
  file << "concentrate = true;\n";
  file << "rankdir = LR;\n";
  for (const auto &node : g.first) {
    auto addr = reinterpret_cast<uintptr_t>(node.get());
    std::string recordInfo = "shape = record";
    if (node->is_param)
      recordInfo += ", color = red";
    recordInfo += "];\n";
    file << "  \"" << addr << "\" [label=\"" << node->label << "| data "
         << std::setprecision(3) << node->data << " | grad " << node->grad
         << "\"" << recordInfo;
    if (node->op != "" && node->op != " ")
      file << "  \"" << addr << node->op << "\" [label=\"" << node->op
           << "\"];\n";
  }
  for (const auto &edge : g.second) {
    auto first_addr = reinterpret_cast<uintptr_t>(edge.first.get());
    auto second_addr = reinterpret_cast<uintptr_t>(edge.second.get());
    // op to parent node
    file << "\"" << first_addr << edge.first->op << "\" -> \"" << first_addr
         << "\";\n";

    // child to op
    file << "\"" << second_addr << "\" -> \"" << first_addr << edge.first->op
         << "\";\n";
  }
  file << "}\n";
  file.close();
}