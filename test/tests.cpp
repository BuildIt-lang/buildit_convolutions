#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "buildit_conv2d.h"
#include <assert.h>
#include <vector>

using namespace torch;
using conv_runtime::TensorT;
using conv_runtime::ConvOptions;

#define N_DIMS 2

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

void test_conv2d(int iw, int ih, int ww, int wh, int batch_size, int in_channels, int out_channels, ConvOptions options) {
    int64_t lo = 0;
    int64_t hi = 100;
    // generate image and kernel
    Tensor torch_input = torch::randint(lo, hi, {batch_size, in_channels, ih, iw}).to(torch::kInt32);
    Tensor torch_weight = torch::randint(lo, hi, {out_channels, in_channels/options.groups, wh, ww}).to(torch::kInt32);
    // get expected output
    int64_t stride_long[] = {(int64_t)options.stride[0], (int64_t)options.stride[1]};
    ArrayRef<int64_t> stride_ref = ArrayRef<int64_t>(stride_long, 2);
    ExpandingArray<N_DIMS> stride_arr = ExpandingArray<N_DIMS>(stride_ref);
    nn::functional::ConvFuncOptions<N_DIMS> torch_options = nn::functional::Conv2dFuncOptions().stride(stride_arr);
    Tensor torch_output = nn::functional::conv2d(torch_input, torch_weight, torch_options);
    std::cout << torch_output << std::endl;
    // get actual output
    TensorT<int> conv_input = {.width = iw, .height = ih, .data = torch_input.data_ptr<int>()};
    TensorT<int> conv_weight = {.width = ww, .height = wh, .data = torch_weight.data_ptr<int>()};
    TensorT<int> conv_output = buildit_conv2d(conv_input, conv_weight, options);
    conv_output.print();
    compare(torch_output, conv_output);
}


int main() {
    // test_conv2d_default_options(5, 5, 2, 3, 1, 1, 1, 1);
    int stride[2] = {3, 2};
    int padding[2] = {0, 0};
    int dilation[2] = {1, 1};
    int groups = 1;
    ConvOptions options = {.stride = stride, .padding = padding, .dilation = dilation, .groups = groups};
    test_conv2d(14, 15, 3, 3, 1, 1, 1, options);
}