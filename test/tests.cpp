#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "buildit_conv2d.h"
#include <assert.h>
#include <array>

using namespace torch;
using conv_runtime::TensorT;

void compare(Tensor expected, TensorT<int> result) {
    assert (result.height == expected.size(2));
    assert (result.width == expected.size(3));
    int w = result.width;
    int h = result.height;
    int* expected_arr = expected.data_ptr<int>();
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            assert(result.data[i*w+j] == expected_arr[i*w+j]);
        }
    }
}

void test_conv2d_default_options(int iw, int ih, int ww, int wh, int batch_size, int in_channels, int out_channels, int groups) {
    int64_t lo = 0;
    int64_t hi = 100;
    Tensor torch_input = torch::randint(lo, hi, {batch_size, in_channels, ih, iw}).to(torch::kInt32);
    Tensor torch_weight = torch::randint(lo, hi, {out_channels, in_channels/groups, wh, ww}).to(torch::kInt32);
    Tensor torch_output = nn::functional::conv2d(torch_input, torch_weight);
    std::cout << torch_output << std::endl;

    TensorT<int> conv_input = {.width = iw, .height = ih, .data = torch_input.data_ptr<int>()};
    TensorT<int> conv_weight = {.width = ww, .height = wh, .data = torch_weight.data_ptr<int>()};
    TensorT<int> conv_output = buildit_conv2d(conv_input, conv_weight, 1, 0, 1, 1);
    conv_output.print();
    compare(torch_output, conv_output);
}

int main() {
    test_conv2d_default_options(5, 5, 2, 3, 1, 1, 1, 1);
}