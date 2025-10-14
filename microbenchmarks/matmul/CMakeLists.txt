cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(matmul_bench STATIC
    bench_runner.cpp
    matmul_registry.cpp
)

target_include_directories(matmul_bench
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(matmul_bench
    PUBLIC
        mlx
)

add_executable(matmul_bench_main bench_entry.cpp)

target_link_libraries(matmul_bench_main
    PRIVATE
        matmul_bench
)
