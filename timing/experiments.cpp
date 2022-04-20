#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "runtime_types.h"
#include "runtime_functions.h"
#include <assert.h>

#include <fstream>
#include "blocks/c_code_generator.h"
#include "blocks/rce.h"
#include "conv_functions/conv2d.h"
#include "pipeline/conv.h"
#include "pipeline/comment_generator.h"

using namespace torch;
using namespace std::chrono;
namespace F = nn::functional;

typedef float conv_t;

typedef conv_runtime::ImageT<conv_t> (*GeneratedFunction) (conv_t* a, conv_t* b);

void compare(Tensor expected, conv_runtime::ImageT<conv_t> result, string test_name, string test_details) {
    std::cout << "Running test: " << test_name << " " << test_details;
    assert(result.batch_size == expected.size(0));
    assert(result.in_channels == expected.size(1));
    assert (result.height == expected.size(2));
    assert (result.width == expected.size(3));
    int w = result.width;
    int h = result.height;
    conv_t* expected_arr = expected.data_ptr<conv_t>();
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float diff = std::abs(result.data[i*w+j] - expected_arr[i*w+j]);
            float threshold = 0.05 * expected_arr[i*w+j];
            if (diff > threshold) {
                std::cout << "expected: " << expected_arr[i*w+j] << ", got: " << result.data[i*w+j] << std::endl;
            }
            assert(diff <= threshold);
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

void time_specialized_conv2d(int iw, int ih, int ww, int wh, int b_sz, int in_ch, int out_ch, int* stride, int* dilation, int* padding, int padding_same, GeneratedFunction func, string test_name) {
    int64_t lo = 0;
    int64_t hi = 100;
    int n_iters = 10;
    Tensor torch_input = torch::randint(lo, hi, {b_sz, in_ch, ih, iw}).to(torch::kFloat32); // use float here if using dilation
    Tensor torch_weight = torch::randint(lo, hi, {out_ch, in_ch, wh, ww}).to(torch::kFloat32);

    // covert params to torch arrays
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_stride = convert_to_expanding_array(stride);
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    torch_options = torch_options.stride(torch_stride).dilation(torch_dilation);
    if (padding_same == 1) {
        torch_options = torch_options.padding(torch::kSame);
    } else {
        ExpandingArray<2> torch_padding = convert_to_expanding_array(padding);
        torch_options = torch_options.padding(torch_padding);
    }

    // time the torch implementation
    conv_runtime::start_timer();
    Tensor torch_output;
    for (int iter = 0; iter < n_iters; iter++) {
        torch_output = F::conv2d(torch_input, torch_weight, torch_options);
    }
    float torch_time = conv_runtime::stop_timer() / n_iters;

    // time buildit generated code
    conv_t* inp_data = torch_input.data_ptr<conv_t>();
    conv_t* kernel_data = torch_weight.data_ptr<conv_t>();
    conv_runtime::start_timer();
    conv_runtime::ImageT<conv_t> spec_conv_output;
    for (int iter = 0; iter < n_iters; iter++) {
        spec_conv_output = func(inp_data, kernel_data);
    }
    float specialized_conv_time = conv_runtime::stop_timer() / n_iters;
    
    // std::cout << torch_output << std::endl;
    // spec_conv_output.print();
    std::cout << "torch_time: " << torch_time << "ms, specialized_conv_time: " << specialized_conv_time << "ms" << ", multiplications: " << spec_conv_output.mult_cnt << std::endl;
    compare(torch_output, spec_conv_output, test_name, "specialized");
}

void run() {
    int n_runs = 8;
    int iw[] = {100, 100, 200, 200, 300, 300, 200, 200, 100};
    int ih[] = {100, 100, 200, 200, 300, 300, 200, 200, 100};
    int kw[] = {10, 10, 10, 10, 10, 10, 10, 10, 20};
    int kh[] = {10, 10, 10, 10, 10, 10, 10, 10, 20};
    int batch_size[] = {10, 20, 10, 20, 10, 20, 10, 10, 10};
    int in_channels[] = {10, 10, 10, 10, 10, 10, 10, 10, 10};
    int out_channels[] = {10, 10, 10, 10, 10, 10, 10, 10, 10};
    int stride[][2] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {5, 5}, {5, 5}};
    int padding[][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {10, 10}, {10, 10}, {10, 10}};
    int dilation[][2] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {10, 10}};
    int padding_same[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::string func_names[] = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"};
    std::string flags = "";
    for (int i = 0; i < n_runs; i++) {
        auto fptr = (GeneratedFunction)pipeline::commented_code_generator::compile_function(
            static_conv2d_with_tiled_loops, flags, iw[i], ih[i], kw[i], kh[i], batch_size[i], in_channels[i], 
            out_channels[i], stride[i], dilation[i], padding[i], padding_same[i]
            );
        time_specialized_conv2d(
            iw[i], ih[i], kw[i], kh[i], batch_size[i], in_channels[i], out_channels[i], 
            stride[i], dilation[i], padding[i], padding_same[i], fptr, func_names[i]
            );
    }
}

int main() {
    run();
}
