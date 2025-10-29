#include "data/mnist.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"

namespace {

void load_csv_with_mmap(const std::string& filename, MNIST_INS& images,
                        MNIST_OUTS& labels) {
  images.clear();
  labels.clear();

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
    return;
  }

  void* mapped =
      ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, /*offset=*/0);
  if (mapped == MAP_FAILED) {
    int err = errno;
    ::close(fd);
    throw std::runtime_error("mmap failed for " + filename + " (" +
                             std::strerror(err) + ")");
  }

  const char* data = static_cast<const char*>(mapped);
  size_t estimated_rows =
      static_cast<size_t>(std::count(data, data + file_size, '\n'));
  if (data[file_size - 1] != '\n') {
    ++estimated_rows;
  }
  if (estimated_rows > 0) {
    images.reserve(estimated_rows);
    labels.reserve(estimated_rows);
  }

  std::vector<float> pixels(784, 0.0f);
  bool have_label = false;
  float label = 0.0f;
  int pixel_index = 0;
  float value = 0.0f;

  try {
    const char* ptr = data;
    const char* end = data + file_size;

    while (ptr < end) {
      unsigned char current = static_cast<unsigned char>(*ptr);
      if (current >= '0' && current <= '9') {
        value = value * 10.0f + static_cast<float>(current - '0');
      }

      const char* next_ptr = ptr + 1;
      bool at_end = (next_ptr >= end);
      char next = at_end ? '\n' : *next_ptr;

      if (next == ',' || next == '\n' || next == '\r' || at_end) {
        if (!have_label) {
          label = value;
          have_label = true;
        } else if (pixel_index < static_cast<int>(pixels.size())) {
          pixels[pixel_index] = value / 255.0f;
          ++pixel_index;
        }

        value = 0.0f;

        if (next == '\n' || next == '\r' || at_end) {
          if (have_label) {
            if (pixel_index < static_cast<int>(pixels.size())) {
              std::fill(pixels.begin() + pixel_index, pixels.end(), 0.0f);
            }
            images.push_back(pixels);
            labels.push_back(label);
          }
          have_label = false;
          pixel_index = 0;
          std::fill(pixels.begin(), pixels.end(), 0.0f);
        }

        ptr = next_ptr;
        if (!at_end) {
          if (*ptr == '\r') {
            ++ptr;
            if (ptr < end && *ptr == '\n') {
              ++ptr;
            }
          } else {
            ++ptr;
          }
        }
        continue;
      }

      ++ptr;
    }

    ::munmap(const_cast<char*>(data), file_size);
    ::close(fd);
  } catch (...) {
    ::munmap(const_cast<char*>(data), file_size);
    ::close(fd);
    throw;
  }
}

}  // namespace

MNIST::MNIST(std::string train_csv_path, std::string test_csv_path) {
  const std::string data_dir = getenv_str("MNIST_DATA_DIR", "data_tmp");
  if (train_csv_path.empty()) {
    train_csv = data_dir + "/mnist_train.csv";
  } else {
    train_csv = train_csv_path;
  }
  if (test_csv_path.empty()) {
    test_csv = data_dir + "/mnist_test.csv";
  } else {
    test_csv = test_csv_path;
  }

  load_data(train_csv, data.train_data, data.train_labels);
  load_data(test_csv, data.test_data, data.test_labels);
}

void MNIST::summary() {
  std::cout << "Train data size: " << data.train_data.size() << std::endl;
  std::cout << "Train labels size: " << data.train_labels.size() << std::endl;
  std::cout << "Test data size: " << data.test_data.size() << std::endl;
  std::cout << "Test labels size: " << data.test_labels.size() << std::endl;
}

void MNIST::load_data(const std::string& csv_filename, MNIST_INS& images,
                      MNIST_OUTS& labels) {
  load_csv_with_mmap(csv_filename, images, labels);
}
