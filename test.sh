
./build.sh "-DENABLE_TESTS=ON"
ctest --test-dir build/release/tests --output-on-failure
exit $?