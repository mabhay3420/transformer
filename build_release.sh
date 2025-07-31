cmake -S . -B build/release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-O3"
cmake --build build/release
