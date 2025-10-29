cmake_minimum_required(VERSION 3.10)

project(TransformersMicrobenchmarks LANGUAGES CXX)

add_subdirectory(matmul)
add_subdirectory(elementwise)
add_subdirectory(csv_loader)
add_subdirectory(mnist_csv)
