#ifndef VIS_HPP
#define VIS_HPP

#include "micrograd.hpp"
#include <set>
#include <utility>
#include <vector>

typedef std::set<const V> snode;
typedef std::set<std::pair<const V, const V>> sedge;
typedef std::pair<snode, sedge> graph;

graph trace(const V root);
void to_dot(const graph &g, std::string &filename);

#endif // VIS_HPP