1. To build the code: `./build.sh`
2. To run the code e2e: `./run.sh`

For debug and address sanitized versions:

1. To build the code: `./build_debug.sh`
2. To run the code e2e: `./run_debug.sh`

## Testing Guidelines

- Always add unit tests for new ops, layers, and training loops under `tests/` using GoogleTest.
- Run tests using `./test.sh`.
- Keep tests small and fast; prefer simple shape cases and exact gradient checks (or finite-difference checks) for new tensor ops.
- Run tests whenever making changes to core autograd/memory/tensor code.

## Tools Usage
1. `ast-grep`: A tool for searching and replacing text in source code.
