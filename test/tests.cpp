#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "buildit_conv2d.h"
#include <assert.h>
#include <vector>

using namespace torch;
using conv_runtime::TensorT;
using conv_runtime::ConvOptions;
namespace F = nn::functional;

#define N_DIMS 2

void compare(Tensor expected, TensorT<int> result) {
    assert (result.height == expected.size(2));
    assert (result.width == expected.size(3));
    int w = result.width;
    int h = result.height;
    float* expected_arr = expected.data_ptr<float>();
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            assert(result.data[i*w+j] == expected_arr[i*w+j]);
        }
    }
    std::cout << "passed\n" << std::endl;
}

ExpandingArray<N_DIMS> convert_to_expanding_array(int* arr) {
    int64_t arr_long[N_DIMS];
    for (int i = 0; i < N_DIMS; i++) {
        arr_long[i] = (int64_t)arr[i];
    }
    ArrayRef<int64_t> arr_ref = ArrayRef<int64_t>(arr_long, N_DIMS);
    ExpandingArray<N_DIMS> expanding_arr = ExpandingArray<N_DIMS>(arr_ref);
    return expanding_arr;
}

void test_conv2d(int iw, int ih, int ww, int wh, int batch_size, int in_channels, int out_channels, ConvOptions conv_options, F::ConvFuncOptions<N_DIMS> torch_options) {
    int64_t lo = 0;
    int64_t hi = 100;
    // generate image and kernel
    Tensor torch_input = torch::randint(lo, hi, {batch_size, in_channels, ih, iw}).to(torch::kFloat); // using float here because dilation doesn't work with int
    Tensor torch_weight = torch::randint(lo, hi, {out_channels, in_channels/conv_options.groups, wh, ww}).to(torch::kFloat);
    // // get expected output
    Tensor torch_output = F::conv2d(torch_input, torch_weight, torch_options);
    // std::cout << torch_output << std::endl;
    // get actual output
    Tensor torch_inp = torch_input.to(torch::kInt32);
    Tensor torch_kernel = torch_weight.to(torch::kInt32);
    TensorT<int> conv_input = {.width = iw, .height = ih, .data = torch_inp.data_ptr<int>()};
    TensorT<int> conv_weight = {.width = ww, .height = wh, .data = torch_kernel.data_ptr<int>()};
    TensorT<int> conv_output = buildit_conv2d(conv_input, conv_weight, conv_options);
    // conv_output.print();
    compare(torch_output, conv_output);
}


int main() {
    // test_conv2d_default_options(5, 5, 2, 3, 1, 1, 1, 1);
    int stride[2] = {2, 1};
    int padding[2] = {0, 0};
    int dilation[2] = {3, 2};
    int groups = 1;
    ConvOptions conv_options = {.stride = stride, .padding = padding, .dilation = dilation, .groups = groups};

    F::ConvFuncOptions<N_DIMS> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<N_DIMS> torch_stride = convert_to_expanding_array(stride);
    ExpandingArray<N_DIMS> torch_dilation = convert_to_expanding_array(dilation);
    torch_options = torch_options.stride(torch_stride).dilation(torch_dilation);
    test_conv2d(10, 10, 2, 3, 1, 1, 1, conv_options, torch_options);
}