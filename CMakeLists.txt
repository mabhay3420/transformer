cmake_minimum_required(VERSION 3.10)

project(Transformers LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(thirdparty)
# Include directories
include_directories(include)
include_directories(thirdparty/json/include)

# Source files
file(GLOB SOURCES src/*.cpp)

# Executable target
add_executable(tformer ${SOURCES})
target_link_libraries(
    tformer
    PRIVATE
    nlohmann_json::nlohmann_json
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(tformer PRIVATE -fsanitize=address)
    target_link_options(tformer PRIVATE -fsanitize=address)
endif()
