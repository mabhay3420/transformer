cmake_minimum_required(VERSION 3.14)

enable_testing()

include(FetchContent)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "GTest not found. Fetching via FetchContent...")
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # For MSVC runtime on Windows; harmless elsewhere
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(tensor_ops_test
  tensor_ops_test.cpp
)

target_include_directories(tensor_ops_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(tensor_ops_test PRIVATE GTest::gtest_main tformer_core)

add_test(NAME tensor_ops_test COMMAND tensor_ops_test)

add_executable(utils_test
  utils_test.cpp
)

target_include_directories(utils_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(utils_test PRIVATE GTest::gtest_main tformer_core)

add_test(NAME utils_test COMMAND utils_test)
