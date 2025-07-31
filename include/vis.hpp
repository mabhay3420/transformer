#ifndef VIS_HPP
#define VIS_HPP

#include "mempool.hpp"
#include "micrograd.hpp"
#include <memory>
#include <set>
#include <utility>
#include <vector>

typedef std::set<const MemPoolIndex> snode;
typedef std::set<std::pair<const MemPoolIndex, const MemPoolIndex>> sedge;
typedef std::tuple<snode, sedge, std::shared_ptr<MemPool<Value>>> graph;

graph trace(const MemPoolIndex root, std::shared_ptr<MemPool<Value>> mem_pool);
void to_dot(const graph &g, std::string &filename);

#endif // VIS_HPP