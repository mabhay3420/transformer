cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(adamw_bench_main bench_entry.cpp)
target_link_libraries(adamw_bench_main
    PRIVATE
        benchmark::benchmark
)
