#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "buildit_conv2d.h"
#include <assert.h>
#include <vector>

using namespace torch;
using conv_runtime::TensorT;
using conv_runtime::ConvOptions;
using conv_runtime::PaddingT;
namespace F = nn::functional;

int default_padding[] = {0, 0};
int default_stride[] = {1, 1};
int default_dilation[] = {1, 1};
int default_groups = 1;
int default_batch_sz = 1;
int default_in_channels = 1;
int default_out_channels = 1;

void compare(Tensor expected, TensorT<int> result, string test_name, string test_details) {
    std::cout << "Running test: " << test_name << " " << test_details;
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
    std::cout << ": PASSED" << std::endl;
}

ExpandingArray<2> convert_to_expanding_array(int* arr) {
    int64_t arr_long[2];
    for (int i = 0; i < 2; i++) {
        arr_long[i] = (int64_t)arr[i];
    }
    ArrayRef<int64_t> arr_ref = ArrayRef<int64_t>(arr_long, 2);
    ExpandingArray<2> expanding_arr = ExpandingArray<2>(arr_ref);
    return expanding_arr;
}

void test_conv2d(int iw, int ih, int ww, int wh, int batch_size, int in_channels, int out_channels, ConvOptions conv_options, F::ConvFuncOptions<2> torch_options, string test_name, string test_details) {
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
    TensorT<int> conv_input = {.batch_size = batch_size, .width = iw, .height = ih, .data = torch_inp.data_ptr<int>()};
    TensorT<int> conv_weight = {.batch_size = batch_size, .width = ww, .height = wh, .data = torch_kernel.data_ptr<int>()};
    TensorT<int> conv_output = buildit_conv2d(conv_input, conv_weight, conv_options);
    // conv_output.print();
    compare(torch_output, conv_output, test_name, test_details);
}

// unit tests

void test_default_options(int batch_sz) {
    ConvOptions conv_options = {.stride = default_stride, .padding = default_padding, .dilation = default_dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    int iw[] = {5, 5, 10, 1};
    int ih[] = {5, 2, 10, 6};
    int ww[] = {3, 2, 6, 1};
    int wh[] = {3, 2, 3, 2};
    int n_tests = 4;
    string details[] = {"nxn image, nxn kernel", "nxn kernel", "nxn image", "1xn image"};
    for (int i = 0; i < n_tests; i++) {
        test_conv2d(iw[i], ih[i], ww[i], wh[i], batch_sz, default_in_channels, default_out_channels, conv_options, torch_options, "default_options", details[i]);
    }
}

void test_stride(int batch_sz) {
    int stride[2] = {2, 1};
    ConvOptions conv_options = {.stride = stride, .padding = default_padding, .dilation = default_dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_stride = convert_to_expanding_array(stride);
    torch_options = torch_options.stride(torch_stride);
    int iw = 10;
    int ih = 8;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, default_in_channels, default_out_channels, conv_options, torch_options, "stride", "");
}

void test_dilation(int batch_sz) {
    int dilation[2] = {3, 2};
    ConvOptions conv_options = {.stride = default_stride, .padding = default_padding, .dilation = dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    torch_options = torch_options.dilation(torch_dilation);
    int iw = 15;
    int ih = 20;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, default_in_channels, default_out_channels, conv_options, torch_options, "dilation", "");
}

void test_stride_dilation(int batch_sz) {
    int dilation[2] = {3, 2};
    int stride[2] = {2, 3};
    ConvOptions conv_options = {.stride = stride, .padding = default_padding, .dilation = dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    ExpandingArray<2> torch_stride = convert_to_expanding_array(stride);
    torch_options = torch_options.dilation(torch_dilation).stride(torch_stride);
    int iw = 15;
    int ih = 20;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, default_in_channels, default_out_channels, conv_options, torch_options, "stride and dilation", "");
}

void test_padding_arr(int batch_sz) {
    int pad_arr[2] = {1, 2};
    PaddingT padding = PaddingT(pad_arr);
    ConvOptions conv_options = {.stride = default_stride, .padding = padding, .dilation = default_dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_padding = convert_to_expanding_array(pad_arr);
    torch_options = torch_options.padding(torch_padding);
    int iw = 5;
    int ih = 5;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, default_in_channels, default_out_channels, conv_options, torch_options, "padding", "arr");
}

void test_padding_same(int batch_sz) {
    char pad_type[] = "same";
    PaddingT padding = PaddingT(pad_type);
    ConvOptions conv_options = {.stride = default_stride, .padding = padding, .dilation = default_dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    torch_options = torch_options.padding(torch::kSame);
    int iw = 5;
    int ih = 5;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, default_in_channels, default_out_channels, conv_options, torch_options, "padding", "same");
}

void test_stride_dilation_padding(int batch_sz) {
    int dilation[2] = {3, 2};
    int stride[2] = {2, 3};
    int pad_arr[2] = {3, 4};
    PaddingT padding = PaddingT(pad_arr);
    ConvOptions conv_options = {.stride = stride, .padding = padding, .dilation = dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    ExpandingArray<2> torch_stride = convert_to_expanding_array(stride);
    ExpandingArray<2> torch_padding = convert_to_expanding_array(pad_arr);
    torch_options = torch_options.dilation(torch_dilation).stride(torch_stride).padding(torch_padding);
    int iw = 15;
    int ih = 20;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, default_in_channels, default_out_channels, conv_options, torch_options, "stride, dilation, padding", "");
}

void test_dilation_padding_same(int batch_sz) {
    int dilation[2] = {3, 2};
    char pad_type[] = "same";
    PaddingT padding = PaddingT(pad_type);
    ConvOptions conv_options = {.stride = default_stride, .padding = padding, .dilation = dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    torch_options = torch_options.dilation(torch_dilation).padding(torch::kSame);
    int iw = 15;
    int ih = 20;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, default_in_channels, default_out_channels, conv_options, torch_options, "dilation, padding", "same");
}

void test_batch_size(int size) {
    std::cout << "Testing batch size: " << size << std::endl;
    test_default_options(size);
    test_stride(size);
    test_dilation(size);
    test_stride_dilation(size);
    test_padding_arr(size);
    test_padding_same(size);
    test_stride_dilation_padding(size);
    test_dilation_padding_same(size);
    std::cout << "Done testing batch size " << size << std::endl;
}


int main() {
    test_default_options(default_batch_sz);
    test_stride(default_batch_sz);
    test_dilation(default_batch_sz);
    test_stride_dilation(default_batch_sz);
    test_padding_arr(default_batch_sz);
    test_padding_same(default_batch_sz);
    test_stride_dilation_padding(default_batch_sz);
    test_dilation_padding_same(default_batch_sz);
    test_batch_size(4);
}