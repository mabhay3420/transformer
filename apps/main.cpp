#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>

#include "bigram.hpp"
#include "bigramnn.hpp"
#include "embednlp.hpp"
#include "mnist.hpp"
#include "utils.hpp"
#include "xormodel_tensors.hpp"

namespace {

struct Command {
  std::string description;
  std::function<void()> action;
};

int run_command(const std::map<std::string, Command>& commands,
                const std::string& name) {
  const auto it = commands.find(name);
  if (it == commands.end()) {
    std::cerr << "Unknown command: " << name
              << "\nAvailable commands:" << std::endl;
    for (const auto& entry : commands) {
      std::cerr << "  " << entry.first << "\t" << entry.second.description
                << std::endl;
    }
    return 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  it->second.action();
  auto end = std::chrono::high_resolution_clock::now();

  using time_unit = std::chrono::duration<double, std::milli>;
  auto duration = std::chrono::duration_cast<time_unit>(end - start);
  std::ofstream fout("time.txt", std::ios::app);
  fout << duration.count() << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::map<std::string, Command> commands = {
      {"xor", {"Tensor autograd XOR", XORWithTensors}},
      {"bigram", {"Bigram language model (Tensor)", BigraLmPT}},
      {"bigram-nn", {"Bigram neural network (Tensor)", BigramNNPT}},
      {"embed", {"Embedded bigram Tensor model", EmbedNLPPT}},
      {"mnist", {"MNIST classifier (Tensor)", MnistDnnPT}},
  };

  if (argc > 1) {
    const std::string arg = argv[1];
    if (arg == "--list" || arg == "-l") {
      for (const auto& entry : commands) {
        std::cout << entry.first << '\t' << entry.second.description
                  << std::endl;
      }
      return 0;
    }
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: tformer [command]\n\nDefaults to 'xor'.\n"
                   "Use --list to see available commands."
                << std::endl;
      return 0;
    }
    return run_command(commands, arg);
  }

  return run_command(commands, "xor");
}
