#ifndef UTILS_HPP
#define UTILS_HPP

#include <iomanip>
#include <ostream>

#include "mempool.hpp"
#include "micrograd.hpp"
#define _print_v(v)                           \
  std::cout << "[ ";                          \
  for (const auto &item : v) {                \
    std::cout << std::setw(3) << item << ","; \
  }                                           \
  std::cout << "]";

#define print_v(v)                                \
  std::cout << "Size: " << v.size() << std::endl; \
  std::cout << "[ ";                              \
  for (const auto &item : v) {                    \
    std::cout << std::setw(3) << item << ",";     \
  }                                               \
  std::cout << "]" << std::endl;

// k_xx so that it doesn't conflict with any other variable names
#define print_v_e(v, e)                                        \
  std::cout << "Size: " << v.size() << std::endl;              \
  std::cout << "First " << e << ": " << std::endl;             \
  std::cout << "[ ";                                           \
  for (size_t k_xx = 0; k_xx < e && k_xx < v.size(); k_xx++) { \
    std::cout << std::setw(3) << v[k_xx] << ",";               \
  }                                                            \
  std::cout << "]" << std::endl;

// k_xx so that it doesn't conflict with any other variable names
#define _print_v_e(v, e)                                       \
  std::cout << "[ ";                                           \
  for (size_t k_xx = 0; k_xx < e && k_xx < v.size(); k_xx++) { \
    std::cout << std::setw(3) << v[k_xx] << ",";               \
  }                                                            \
  std::cout << "]";

#define print_vv(vv)                                                           \
  std::cout << "Size: " << vv.size() << "x" << (vv.empty() ? 0 : vv[0].size()) \
            << std::endl;                                                      \
  std::cout << "[ " << std::endl;                                              \
  for (const auto &v : vv) {                                                   \
    _print_v(v);                                                               \
    std::cout << "," << std::endl;                                             \
  }                                                                            \
  std::cout << "] " << std::endl;
#endif  // UTILS_HPP

inline std::ostream &operator<<(std::ostream &os, const std::vector<int> &v) {
  os << "[ ";
  for (const auto &item : v) {
    os << std::setw(3) << item << ",";
  }
  os << "]";
  return os;
}
inline std::ostream &operator<<(std::ostream &os, const std::vector<float> &v) {
  os << "[ ";
  for (const auto &item : v) {
    os << std::setw(3) << item << ",";
  }
  os << "]";
  return os;
}
inline std::ostream &operator<<(std::ostream &os, const std::vector<V> &v) {
  os << "[ ";
  for (const auto &item : v) {
    os << std::setw(3) << item->data << ",";
  }
  os << "]";
  return os;
}

struct Indent {
  int level = 0;
  Indent() : level(0) {}
  Indent(int level) : level(level) {}
  friend std::ostream &operator<<(std::ostream &os, const Indent &i) {
    for (int j = 0; j < i.level; j++) {
      os << "  ";
    }
    return os;
  }
};

void dumpJson(json &j, std::string filename);
void dumpJson(json &j, const char *filename);
void dumpValues(std::vector<V> values, const char *filename);
void dumpValues(std::vector<std::pair<V, V>> values, const char *filename);
void dumpValues(std::vector<std::pair<V, V>> values, std::string filename);
void dumpValues(std::vector<V> values, std::string filename);
void dumpValues(std::vector<std::pair<V, V>> values, std::string filename);
void dumpMemPoolEntries(
    std::vector<std::pair<MemPoolIndex, MemPoolIndex>> entries,
    MemPool<Value> *mem_pool, std::string filename);
void dumpMemPoolEntries(
    std::vector<std::pair<MemPoolIndex, MemPoolIndex>> entries,
    MemPool<Value> *mem_pool, const char *filename);
void dumpMemPoolEntries(std::vector<MemPoolIndex> entries,
                        MemPool<Value> *mem_pool, std::string filename);

void dumpMemPoolEntries(std::vector<MemPoolIndex> entries,
                        MemPool<Value> *mem_pool, const char *filename);

float get_random_float(float min, float max);