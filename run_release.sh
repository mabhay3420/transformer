cmake -S . -B build/release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/release
./build/release/tformer
