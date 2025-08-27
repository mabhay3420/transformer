#ifndef VIS_HPP
#define VIS_HPP

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "mempool.hpp"
#include "micrograd.hpp"

typedef std::set<const MemPoolIndex> snode;
typedef std::set<std::pair<const MemPoolIndex, const MemPoolIndex>> sedge;
typedef std::tuple<snode, sedge, MemPool<Value> *> graph;

graph trace(const MemPoolIndex root, MemPool<Value> *mem_pool);
void to_dot(const graph &g, std::string &filename);

#endif  // VIS_HPP