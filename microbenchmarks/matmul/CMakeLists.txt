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

find_library(ACCELERATE_FRAMEWORK Accelerate)
if(NOT ACCELERATE_FRAMEWORK)
    message(FATAL_ERROR "Accelerate framework not found.")
endif()

set(CBLAS_LIB ${ACCELERATE_FRAMEWORK})
target_compile_definitions(matmul_bench
    PUBLIC
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
)

target_link_libraries(matmul_bench
    PUBLIC
        mlx
        ${CBLAS_LIB}
)

add_executable(matmul_bench_main bench_entry.cpp)

target_link_libraries(matmul_bench_main
    PRIVATE
        matmul_bench
        ${CBLAS_LIB}
)
