cmake_minimum_required(VERSION 3.10)

project(Transformers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(thirdparty)
add_subdirectory(microbenchmarks)

set(TFORMER_CORE_SOURCES
    lib/core/learning_rate.cpp
    lib/core/tensor.cpp
    lib/data/mnist.cpp
    lib/data/sampler.cpp
    lib/data/swedish_auto.cpp
    lib/data/tokenizer.cpp
    lib/data/text.cpp
    lib/models/bigram.cpp
    lib/models/bigramnn.cpp
    lib/models/embednlp.cpp
    lib/models/mnist.cpp
    lib/models/nlp.cpp
    lib/models/xormodel_tensors.cpp
    lib/nn/nn.cpp
    lib/train/language_utils.cpp
    lib/utils/probs.cpp
    lib/utils/utils.cpp
)

add_library(tformer_core STATIC ${TFORMER_CORE_SOURCES})
target_include_directories(
    tformer_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)
target_link_libraries(
    tformer_core
    PUBLIC
    nlohmann_json::nlohmann_json
    mlx
    mlxdata
    bxzstr
)

add_executable(tformer apps/main.cpp)
target_link_libraries(tformer PRIVATE tformer_core)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
    target_compile_options(tformer_core PRIVATE -fsanitize=address)
    target_link_options(tformer_core PRIVATE -fsanitize=address)
    target_compile_options(tformer PRIVATE -fsanitize=address)
    target_link_options(tformer PRIVATE -fsanitize=address)
endif()

option(ENABLE_TESTS "Build unit tests" ON)
if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
else()
    message(WARNING "Doxygen not found. Documentation target disabled.")
endif()
