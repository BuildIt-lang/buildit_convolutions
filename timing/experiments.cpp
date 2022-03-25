#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "buildit_conv2d.h"
#include "specialized_test_code.h"
#include "runtime_functions.h"
#include <assert.h>


using namespace torch;
using conv_runtime::ConvOptions;
using conv_runtime::PaddingT;
using conv_runtime::ImageT;
using conv_runtime::KernelT;
using namespace std::chrono;
namespace F = nn::functional;

void compare(Tensor expected, ImageT<int> result, string test_name, string test_details) {
    std::cout << "Running test: " << test_name << " " << test_details;
    assert(result.batch_size == expected.size(0));
    assert(result.in_channels == expected.size(1));
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

void time_conv2d(int iw, int ih, int ww, int wh, int batch_size, int in_channels, int out_channels, ConvOptions conv_options, F::ConvFuncOptions<2> torch_options, string test_name, string test_details) {
    int64_t lo = 0;
    int64_t hi = 100;
    int n_iters = 100;
    // generate image and kernel
    Tensor torch_input = torch::randint(lo, hi, {batch_size, in_channels, ih, iw}).to(torch::kFloat); // using float here because dilation doesn't work with int
    Tensor torch_weight = torch::randint(lo, hi, {out_channels, in_channels/conv_options.groups, wh, ww}).to(torch::kFloat);
    
    conv_runtime::start_timer();
    for (int iter = 0; iter < n_iters; iter++) {
        Tensor torch_output = F::conv2d(torch_input, torch_weight, torch_options);
    }
    float torch_time = conv_runtime::stop_timer() / n_iters;

    Tensor torch_inp = torch_input.to(torch::kInt32);
    Tensor torch_kernel = torch_weight.to(torch::kInt32);
    ImageT<int> conv_input = {.batch_size = batch_size, .in_channels = in_channels, .width = iw, .height = ih, .data = torch_inp.data_ptr<int>()};
    KernelT<int> conv_weight = {.out_channels = out_channels, .in_channels = in_channels, .width = ww, .height = wh, .data = torch_kernel.data_ptr<int>()};

    conv_runtime::start_timer();
    for (int iter = 0; iter < n_iters; iter++) {
        ImageT<int> conv_output = buildit_conv2d(conv_input, conv_weight, conv_options);
    }
    float conv_time = conv_runtime::stop_timer() / n_iters;
    
    std::cout << "torch_time: " << torch_time << "ms, conv_time: " << conv_time << "ms" << std::endl;
}

void run() {
    int n_runs = 3;
    int iw[] = {10, 100, 1000};
    int ih[] = {10, 100, 100};
    int kw[] = {3, 10, 10};
    int kh[] = {3, 10, 10};
    int batch_size[] = {10, 10, 10};
    int in_channels[] = {10, 10, 10};
    int out_channels[] = {1, 1, 1};
    int stride[][2] = {{1, 1}, {1, 1}, {1, 1}};
    int padding[][2] = {{0, 0}, {0, 0}, {0, 0}};
    int dilation[][2] = {{1, 1}, {1, 1}, {1, 1}};
    int groups[] = {1, 1, 1};

    for (int i = 0; i < n_runs; i++) {
        ConvOptions conv_options = {.stride = stride[i], .padding = padding[i], .dilation = dilation[i], .groups = groups[i]};
        F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
        ExpandingArray<2> torch_stride = convert_to_expanding_array(stride[i]);
        ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation[i]);
        ExpandingArray<2> torch_padding = convert_to_expanding_array(padding[i]);
        torch_options = torch_options.stride(torch_stride).padding(torch_padding).dilation(torch_dilation);
        time_conv2d(iw[i], ih[i], kw[i], kh[i], batch_size[i], in_channels[i], out_channels[i], conv_options, torch_options, "test", "test");
    }
}

int main() {
    run();
}