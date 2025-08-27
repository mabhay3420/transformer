CMAKE_ARGS=$1
cmake -S . -B build/release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-O3" ${CMAKE_ARGS}
cmake --build build/release
rm -rf compile_commands.json
ln -s build/release/compile_commands.json compile_commands.json
