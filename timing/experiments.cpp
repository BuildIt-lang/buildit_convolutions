#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "runtime_types.h"
#include "runtime_functions.h"
#include <assert.h>

#include <fstream>
#include "blocks/c_code_generator.h"
#include "blocks/rce.h"
#include "conv_functions/conv_nd.h"
#include "pipeline/conv_code_generator.h"
#include "timing_code.h"

using namespace torch;
using namespace std::chrono;
namespace F = nn::functional;

typedef float conv_t;

typedef conv_runtime::ImageT<conv_t> (*GeneratedFunction) (conv_t* a, conv_t* b);

void compare(Tensor expected, conv_runtime::ImageT<conv_t> result, string test_name, string test_details) {
    std::cout << "Running test: " << test_name << " " << test_details;
    assert(result.batch_size == expected.size(0));
    assert(result.in_channels == expected.size(1));
    assert (result.dims[0] == expected.size(2));
    assert (result.dims[1] == expected.size(3));
    int w = result.dims[1];
    int h = result.dims[0];
    conv_t* expected_arr = expected.data_ptr<conv_t>();
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float diff = std::abs(result.data[i*w+j] - expected_arr[i*w+j]);
            float threshold = 0.001;
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
    int n_iters = 50;
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
    Tensor torch_output;
    conv_runtime::start_timer();
    for (int iter = 0; iter < n_iters; iter++) {
        torch_output = F::conv2d(torch_input, torch_weight, torch_options);
    }
    float torch_time = conv_runtime::stop_timer() / n_iters;
    // time buildit generated code
    conv_t* inp_data = torch_input.data_ptr<conv_t>();
    conv_t* kernel_data = torch_weight.data_ptr<conv_t>();
    conv_runtime::ImageT<conv_t> spec_conv_output;
    conv_runtime::start_timer();
    for (int iter = 0; iter < n_iters; iter++) {
        spec_conv_output = func(inp_data, kernel_data);
    }
    float specialized_conv_time = conv_runtime::stop_timer() / n_iters;
    
    // std::cout << torch_output << std::endl;
    // spec_conv_output.print();
    std::cout << "torch_time: " << torch_time << "s, specialized_conv_time: " << specialized_conv_time << "s" << ", multiplications: " << spec_conv_output.mult_cnt << std::endl;
    compare(torch_output, spec_conv_output, test_name, "specialized");
}

void run() {
    int n_runs = 9;
    int iw[] = {100, 100, 200, 200, 300, 300, 200, 200, 128};
    int ih[] = {100, 100, 200, 200, 300, 300, 200, 200, 128};
    int kw[] = {10, 10, 10, 10, 10, 10, 10, 10, 5};
    int kh[] = {10, 10, 10, 10, 10, 10, 10, 10, 5};
    int batch_size[] = {10, 20, 10, 20, 10, 20, 10, 10, 32};
    int in_channels[] = {10, 10, 10, 10, 10, 10, 10, 10, 64};
    int out_channels[] = {10, 10, 10, 10, 10, 10, 10, 10, 16};
    int stride[][2] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {5, 5}, {1, 1}};
    int padding[][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {10, 10}, {10, 10}, {0, 0}};
    int dilation[][2] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    int padding_same[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::string func_names[] = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"};
    GeneratedFunction functions[] = {&f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9};
    for (int i = 0; i < n_runs; i++) {
        time_specialized_conv2d(iw[i], ih[i], kw[i], kh[i], batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], functions[i], func_names[i]);
    }
}

int main() {
    run();
}
