# Resources
1. Dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
2. Video: https://www.youtube.com/watch?v=kCc8FmEb1nY


# Guidelines
where possible:
    - use std library
    - use auto

# Running

- `./run.sh` builds the project and executes the default `xor-pt` command, which trains the XOR task using the PyTorch-style `nn` modules and tensor autograd.
- `./build/release/tformer --list` shows all available entrypoints. Use `./build/release/tformer <command>` or `./run.sh <command>` to run an alternative experiment (e.g. `bigram`, `mnist`).
