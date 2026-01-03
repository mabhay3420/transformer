cmake_minimum_required(VERSION 3.10)

project(TransformersMicrobenchmarks LANGUAGES CXX)

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )
  FetchContent_MakeAvailable(benchmark)
endif()

add_subdirectory(matmul)
add_subdirectory(elementwise)
add_subdirectory(csv_loader)
add_subdirectory(mnist_csv)
add_subdirectory(adamw)
