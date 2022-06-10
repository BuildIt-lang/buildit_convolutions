#!/bin/bash
make
./build/sample4
cd test
cmake --build build
./build/buildit_conv_test
cd ..
