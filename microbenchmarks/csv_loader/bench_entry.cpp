#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "dataloader.hpp"
#include "mlx/data/core/CSVReader.h"

namespace {

using clock = std::chrono::steady_clock;

struct Config {
  std::string file;
  int max_lines = -1;
  int iterations = 5;
};

Config parse_flags(int argc, char** argv) {
  Config cfg;
  const char* env_dir = std::getenv("MNIST_DATA_DIR");
  std::string data_dir =
      (env_dir && env_dir[0] != '\0') ? std::string(env_dir) : "data_tmp";
  cfg.file = data_dir + "/mnist_train.csv";
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("FILE=", 0) == 0) {
      cfg.file = arg.substr(5);
    } else if (arg.rfind("LINES=", 0) == 0) {
      cfg.max_lines = std::stoi(arg.substr(6));
    } else if (arg.rfind("ITER=", 0) == 0) {
      cfg.iterations = std::stoi(arg.substr(5));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(1);
    }
  }
  if (cfg.iterations < 1) cfg.iterations = 1;
  return cfg;
}

struct IterationResult {
  double millis = 0.0;
  size_t rows = 0;
  size_t cols = 0;
  double checksum = 0.0;
};

std::string read_file(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open CSV file: " + filename);
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

MNIST_BATCH load_csv_batch(const std::string& csv_filename, int max_lines) {
  MNIST_INS images;
  MNIST_OUTS labels;

  std::string csv_contents = read_file(csv_filename);
  if (csv_contents.empty()) {
    return {std::move(images), std::move(labels)};
  }

  size_t total_rows = static_cast<size_t>(
      std::count(csv_contents.begin(), csv_contents.end(), '\n'));
  if (!csv_contents.empty() && csv_contents.back() != '\n') {
    total_rows += 1;
  }
  if (total_rows == 0) {
    return {std::move(images), std::move(labels)};
  }
  if (max_lines > 0) {
    total_rows = std::min<size_t>(total_rows, static_cast<size_t>(max_lines));
  }
  images.reserve(total_rows);
  labels.reserve(total_rows);

  auto csv_stream =
      std::make_shared<std::istringstream>(std::move(csv_contents));
  mlx::data::core::CSVReader reader(csv_stream, ',', '"');

  size_t loaded = 0;
  while (loaded < total_rows) {
    auto row = reader.next();
    if (row.empty()) break;
    if (row.size() < 2) {
      throw std::runtime_error("CSV row must contain label and pixels");
    }

    labels.push_back(
        static_cast<float>(std::strtol(row.front().c_str(), nullptr, 10)));

    MNIST_IN pixels(row.size() - 1);
    for (size_t i = 1; i < row.size(); ++i) {
      pixels[i - 1] = std::strtof(row[i].c_str(), nullptr) / 255.0f;
    }
    images.push_back(std::move(pixels));
    ++loaded;
  }

  return {std::move(images), std::move(labels)};
}

IterationResult run_once(const Config& cfg) {
  IterationResult result;
  auto start = clock::now();
  auto batch = load_csv_batch(cfg.file, cfg.max_lines);
  auto end = clock::now();
  result.millis =
      std::chrono::duration<double, std::milli>(end - start).count();
  result.rows = batch.first.size();
  result.cols = result.rows > 0 ? batch.first.front().size() : 0;

  double checksum = 0.0;
  for (const auto& row : batch.first) {
    checksum += std::accumulate(row.begin(), row.end(), 0.0);
  }
  checksum += std::accumulate(batch.second.begin(), batch.second.end(), 0.0);
  result.checksum = checksum;
  static volatile double sink = 0.0;
  sink += checksum;
  return result;
}

}  // namespace

int main(int argc, char** argv) try {
  auto cfg = parse_flags(argc, argv);

  std::cout << "csv_loader benchmark" << std::endl;
  std::cout << "  file       : " << cfg.file << std::endl;
  std::cout << "  max lines  : " << cfg.max_lines << std::endl;
  std::cout << "  iterations : " << cfg.iterations << std::endl;

  // Warm-up iteration (not recorded) to mitigate cold-start effects.
  auto warm_up = run_once(cfg);

  std::vector<double> times_ms;
  times_ms.reserve(static_cast<size_t>(cfg.iterations));
  size_t rows = warm_up.rows;
  size_t cols = warm_up.cols;
  for (int i = 0; i < cfg.iterations; ++i) {
    auto iter = run_once(cfg);
    rows = iter.rows;
    cols = iter.cols;
    times_ms.push_back(iter.millis);
  }

  if (times_ms.empty()) {
    std::cout << "No iterations recorded." << std::endl;
    return 0;
  }

  auto [min_it, max_it] = std::minmax_element(times_ms.begin(), times_ms.end());
  double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
                  static_cast<double>(times_ms.size());

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Rows loaded : " << rows
            << (cols > 0 ? " (pixels per row " + std::to_string(cols) + ")"
                         : "")
            << std::endl;
  std::cout << "Latency ms  : min=" << *min_it << " avg=" << avg_ms
            << " max=" << *max_it << std::endl;
  if (rows > 0 && avg_ms > 0.0) {
    double rows_per_sec = (rows * 1000.0) / avg_ms;
    std::cout << "Throughput  : " << rows_per_sec << " rows/s" << std::endl;
  }
  std::cout.unsetf(std::ios::floatfield);

  return 0;
} catch (const std::exception& ex) {
  std::cerr << "csv_loader benchmark failed: " << ex.what() << std::endl;
  return 1;
}
