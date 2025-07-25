#!/bin/bash

# Remove the 'build' directory if it exists
rm -rf build

# Create a new 'build' directory
mkdir build
#rm -r
# Change into the 'build' directory
cd build

# Run cmake with the specified options
cmake .. -DCMAKE_PREFIX_PATH=/home/user/libtorch -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_STANDARD=17

# Run make and redirect both stdout and stderr to 'output.txt'
make > output.txt 2>&1

# Run the binary
./DDPG