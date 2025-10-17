#!/bin/bash

# Generate Doxygen documentation
# Assumes build/release exists and is configured

cmake --build build/release --target doc