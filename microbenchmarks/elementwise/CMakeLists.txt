cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(elementwise_bench_main bench_entry.cpp)

target_include_directories(elementwise_bench_main
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
