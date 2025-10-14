cmake_minimum_required(VERSION 3.10)

project(Transformers LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(thirdparty)
# Include directories
include_directories(include)
include_directories(thirdparty/json/include)
include_directories(thirdparty)
# include_directories(thirdparty/mlx)
# include_directories(thirdparty/mlx-data)

add_subdirectory(microbenchmarks)

# Source files
file(GLOB SOURCES src/*.cpp)

# Executable target
add_executable(tformer ${SOURCES})
target_link_libraries(
    tformer
    PRIVATE
    nlohmann_json::nlohmann_json
    mlx
    mlxdata
    bxzstr
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
    target_compile_options(tformer PRIVATE -fsanitize=address)
    target_link_options(tformer PRIVATE -fsanitize=address)
endif()

# Testing
option(ENABLE_TESTS "Build unit tests" ON)
if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
