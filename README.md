# Convolutions with BuildIt

To generate code for one of the programs in `./samples` run the following command from the root directory:

```
make && ./build/sample1
```

## Generating code for conv2d and testing

To generate the code for the `conv2d` function run:

```
make && ./build/sample1
```
The generated code should be in `./generated_code/specialized_test_code.h`

The `./test` directory contains testing code which uses PyTorch. 
If you don't have libtorch downloaded you can get it with the following command:
```
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
To setup the testing code run the following from the root directory of this repo:
```
cd test
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```
After the setup you can run the testing from the `./test/` dir with the following command:
```
cmake --build build && ./build/buildit_conv_test
```
