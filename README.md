# Convolutions with BuildIt

To generate code for one of the programs in `./samples` run the following command from the root directory:

```
make && ./build/sample1
```

The generated code should be in `./generated_code/`.

## Testing and code performance

The `./test` directory contains testing code which uses PyTorch. 
If you don't have libtorch downloaded you can get it with the following command:
```
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
To set up the testing code run the following from the root directory of this repo:
```
cd test
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```
After the setup you can test the code from the root directory with the following command:
```
bash run_test.sh
```

To compare the performance of the generated code with PyTorch run the following:
```
bash run_timing.sh
```

