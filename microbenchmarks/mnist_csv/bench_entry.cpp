#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mnist_csv_bench {

using BenchClock = std::chrono::steady_clock;

using MNIST_IN = std::vector<float>;
using MNIST_OUT = float;
using MNIST_INS = std::vector<MNIST_IN>;
using MNIST_OUTS = std::vector<MNIST_OUT>;
using MNIST_BATCH = std::pair<MNIST_INS, MNIST_OUTS>;

struct MnistDataset {
  MNIST_INS train_data;
  MNIST_OUTS train_labels;
  MNIST_INS test_data;
  MNIST_OUTS test_labels;
};

struct Config {
  int max_lines = -1;
  int iterations = 5;
};

struct IterationResult {
  double millis_fstream = 0.0;
  double millis_mmap = 0.0;
  size_t train_rows = 0;
  size_t test_rows = 0;
  double checksum = 0.0;
};

Config parse_flags(int argc, char** argv) {
  if (argc > 1) {
    std::cerr << "mnist_csv benchmark does not accept command line overrides; "
                 "use defaults instead."
              << std::endl;
    std::exit(1);
  }
  return {};
}

std::string resolve_train_path() { return "data_tmp/mnist_train.csv"; }

std::string resolve_test_path() { return "data_tmp/mnist_test.csv"; }

MNIST_BATCH load_csv_with_fstream(const std::string& filename, int max_lines) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open MNIST CSV file: " + filename);
  }

  MNIST_INS images;
  MNIST_OUTS labels;
  std::string line;

  float label;
  std::vector<float> image(784);
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string token;

    std::getline(ss, token, ',');
    label = std::stoi(token) / 1.0f;

    // get next remaining tokens
    size_t i = 0;
    while (std::getline(ss, token, ',')) {
      image[i++] = std::stoi(token) / 1.0f;
    }
    images.push_back(image);
    labels.push_back(label);
  }

  return {std::move(images), std::move(labels)};
}

MNIST_BATCH load_csv_with_mmap(const std::string& filename, int max_lines) {
  int fd = ::open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Failed to open MNIST CSV file for mmap: " +
                             filename + " (" + std::strerror(errno) + ")");
  }

  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    int err = errno;
    ::close(fd);
    throw std::runtime_error("fstat failed for " + filename + " (" +
                             std::strerror(err) + ")");
  }
  size_t file_size = static_cast<size_t>(st.st_size);
  if (file_size == 0) {
    ::close(fd);
    return {};
  }

  void* mapped =
      ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, /*offset=*/0);
  if (mapped == MAP_FAILED) {
    int err = errno;
    ::close(fd);
    throw std::runtime_error("mmap failed for " + filename + " (" +
                             std::strerror(err) + ")");
  }

  MNIST_INS images;
  MNIST_OUTS labels;
  int processed = 0;

  const char* data = static_cast<const char*>(mapped);
  const char* ptr = data;
  const char* end = data + file_size;

  std::vector<float> image(784, 0.0f);
  bool label_set = false;
  float label;
  int image_idx = 0;
  float val = 0.0f;
  bool last_zero = false;

  try {
    for (size_t i = 0; i < file_size; ++i) {
      val = (val * 10.0f) + (data[i] - '0');
      // time to write data
      if (data[i + 1] == ',' || data[i + 1] == '\n' || data[i + 1] == '\r') {
        if (label_set) {
          image[image_idx++] = val;
        } else {
          label = val;
          label_set = true;
        }

        // time to write data to main memory
        if (data[i + 1] == '\n' || data[i + 1] == '\r') {
          images.push_back(image);
          labels.push_back(label);
          image_idx = 0;
          label_set = false;
        }

        // move ahead
        i++;
        val = 0.0f;
      }
    }

    ::munmap(const_cast<char*>(data), file_size);
    ::close(fd);
  } catch (...) {
    ::munmap(const_cast<char*>(data), file_size);
    ::close(fd);
    throw;
  }

  return {std::move(images), std::move(labels)};
}

void verify_batches_match(const MNIST_BATCH& a, const MNIST_BATCH& b,
                          const char* tag) {
  if (a.first.size() != b.first.size() || a.second.size() != b.second.size()) {
    throw std::runtime_error(std::string("MNIST batch size mismatch (") + tag +
                             ")");
  }
  const float tol = 1e-6f;
  for (size_t row = 0; row < a.first.size(); ++row) {
    const auto& img_a = a.first[row];
    const auto& img_b = b.first[row];
    if (img_a.size() != img_b.size()) {
      throw std::runtime_error(std::string("MNIST image width mismatch (") +
                               tag + ")");
    }
    for (size_t col = 0; col < img_a.size(); ++col) {
      if (std::fabs(img_a[col] - img_b[col]) > tol) {
        // print the differing value and col
        std::cout << "Differing value: A: " << img_a[col]
                  << ", B: " << img_b[col] << " at row " << row << " at col "
                  << col << std::endl;
        throw std::runtime_error(std::string("MNIST image value mismatch (") +
                                 tag + ")");
      }
    }
  }
  for (size_t i = 0; i < a.second.size(); ++i) {
    if (std::fabs(a.second[i] - b.second[i]) > tol) {
      throw std::runtime_error(std::string("MNIST label mismatch (") + tag +
                               ")");
    }
  }
}

IterationResult run_once(const Config& cfg) {
  IterationResult result;

  const auto fstream_start = BenchClock::now();
  auto train_fstream =
      load_csv_with_fstream(resolve_train_path(), cfg.max_lines);
  auto test_fstream = load_csv_with_fstream(resolve_test_path(), cfg.max_lines);
  const auto fstream_end = BenchClock::now();
  result.millis_fstream =
      std::chrono::duration<double, std::milli>(fstream_end - fstream_start)
          .count();

  const auto mmap_start = BenchClock::now();
  auto train_mmap = load_csv_with_mmap(resolve_train_path(), cfg.max_lines);
  auto test_mmap = load_csv_with_mmap(resolve_test_path(), cfg.max_lines);
  const auto mmap_end = BenchClock::now();
  result.millis_mmap =
      std::chrono::duration<double, std::milli>(mmap_end - mmap_start).count();

  verify_batches_match(train_fstream, train_mmap, "train");
  verify_batches_match(test_fstream, test_mmap, "test");

  MnistDataset dataset;
  dataset.train_data = std::move(train_fstream.first);
  dataset.train_labels = std::move(train_fstream.second);
  dataset.test_data = std::move(test_fstream.first);
  dataset.test_labels = std::move(test_fstream.second);

  result.train_rows = dataset.train_data.size();
  result.test_rows = dataset.test_data.size();

  double checksum = 0.0;
  for (const auto& row : dataset.train_data) {
    checksum += std::accumulate(row.begin(), row.end(), 0.0);
  }
  checksum += std::accumulate(dataset.train_labels.begin(),
                              dataset.train_labels.end(), 0.0);
  for (const auto& row : dataset.test_data) {
    checksum += std::accumulate(row.begin(), row.end(), 0.0);
  }
  checksum += std::accumulate(dataset.test_labels.begin(),
                              dataset.test_labels.end(), 0.0);
  result.checksum = checksum;

  static volatile double sink = 0.0;
  sink += checksum;

  return result;
}

void summarize_results(const std::vector<IterationResult>& results) {
  if (results.empty()) return;

  double sum_fstream = 0.0;
  double sum_sq_fstream = 0.0;
  double sum_mmap = 0.0;
  double sum_sq_mmap = 0.0;
  for (const auto& res : results) {
    sum_fstream += res.millis_fstream;
    sum_sq_fstream += res.millis_fstream * res.millis_fstream;
    sum_mmap += res.millis_mmap;
    sum_sq_mmap += res.millis_mmap * res.millis_mmap;
  }

  const double count = static_cast<double>(results.size());
  const double avg_fstream = sum_fstream / count;
  double var_fstream = (sum_sq_fstream / count) - (avg_fstream * avg_fstream);
  if (var_fstream < 0.0) var_fstream = 0.0;
  const double stddev_fstream = std::sqrt(var_fstream);

  const double avg_mmap = sum_mmap / count;
  double var_mmap = (sum_sq_mmap / count) - (avg_mmap * avg_mmap);
  if (var_mmap < 0.0) var_mmap = 0.0;
  const double stddev_mmap = std::sqrt(var_mmap);

  const IterationResult& last = results.back();
  std::cout << "\nSummary\n";
  std::cout << "  train rows : " << last.train_rows << '\n';
  std::cout << "  test rows  : " << last.test_rows << '\n';
  std::cout << "  checksum   : " << last.checksum << '\n';
  std::cout << "  fstream latency : " << avg_fstream << " ms (stddev "
            << stddev_fstream << " ms)\n";
  std::cout << "  mmap latency    : " << avg_mmap << " ms (stddev "
            << stddev_mmap << " ms)\n";
}

}  // namespace mnist_csv_bench

int main(int argc, char** argv) try {
  auto cfg = mnist_csv_bench::parse_flags(argc, argv);
  const std::string train_path = mnist_csv_bench::resolve_train_path();
  const std::string test_path = mnist_csv_bench::resolve_test_path();

  std::cout << "mnist_csv benchmark\n";
  std::cout << "  train file : " << train_path << '\n';
  std::cout << "  test file  : " << test_path << '\n';
  std::cout << "  max lines  : " << cfg.max_lines << '\n';
  std::cout << "  iterations : " << cfg.iterations << '\n';
  std::cout << std::fixed << std::setprecision(4);

  std::vector<mnist_csv_bench::IterationResult> results;
  results.reserve(static_cast<size_t>(cfg.iterations));
  for (int iter = 0; iter < cfg.iterations; ++iter) {
    auto res = mnist_csv_bench::run_once(cfg);
    results.push_back(res);
    std::cout << "iteration " << (iter + 1) << ": fstream "
              << res.millis_fstream << " ms, mmap " << res.millis_mmap
              << " ms, checksum " << res.checksum << '\n';
  }

  mnist_csv_bench::summarize_results(results);
  return 0;
} catch (const std::exception& ex) {
  std::cerr << "mnist_csv benchmark failed: " << ex.what() << std::endl;
  return 1;
} catch (...) {
  std::cerr << "mnist_csv benchmark failed: unknown error" << std::endl;
  return 1;
}
