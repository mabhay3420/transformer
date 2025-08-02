#include "utils.hpp"
#include "mempool.hpp"
#include <fstream>
#include <iostream>
#include <memory>

void dumpJson(json &j, std::string filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cout << "Could not open file for writing: " << filename << std::endl;
    throw std::runtime_error("FILE_NOT_FOUND");
  }
  file << j.dump(4) << std::endl;
  file.close();
}
void dumpJson(json &j, const char *filename) {
  std::string str(filename);
  dumpJson(j, str);
}

void dumpValues(std::vector<V> values, std::string filename) {
  json j = values;
  dumpJson(j, filename);
}
void dumpValues(std::vector<V> values, const char *filename) {
  std::string str(filename);
  dumpValues(values, str);
}
void dumpValues(std::vector<std::pair<V, V>> values, std::string filename) {
  json j = values;
  dumpJson(j, filename);
}
void dumpValues(std::vector<std::pair<V, V>> values, const char *filename) {
  std::string str(filename);
  dumpValues(values, str);
}

void dumpMemPoolEntries(
    std::vector<std::pair<MemPoolIndex, MemPoolIndex>> entries,
    std::shared_ptr<MemPool<Value>> mem_pool, std::string filename) {
  std::vector<Value *> values;
  for (auto e : entries) {
    values.push_back(mem_pool->get(e.first));
    values.push_back(mem_pool->get(e.second));
  }
  json j = values;
  dumpJson(j, filename);
}

void dumpMemPoolEntries(
    std::vector<std::pair<MemPoolIndex, MemPoolIndex>> entries,
    std::shared_ptr<MemPool<Value>> mem_pool, const char *filename) {
  std::string str(filename);
  dumpMemPoolEntries(entries, mem_pool, str);
}
void dumpMemPoolEntries(std::vector<MemPoolIndex> entries,
                        std::shared_ptr<MemPool<Value>> mem_pool,
                        std::string filename) {
  std::vector<Value *> values;
  for (auto e : entries) {
    values.push_back(mem_pool->get(e));
  }
  json j = values;
  dumpJson(j, filename);
}

void dumpMemPoolEntries(std::vector<MemPoolIndex> entries,
                        std::shared_ptr<MemPool<Value>> mem_pool,
                        const char *filename) {
  std::string str(filename);
  dumpMemPoolEntries(entries, mem_pool, str);
}

float get_random_float(float min, float max) {
  return (float)rand() / RAND_MAX * (max - min) + min;
};