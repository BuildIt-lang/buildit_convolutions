#include <iostream>
#include <torch/torch.h>
#include "buildit_conv2d.h"
#include "test_types.h"
#include <assert.h>
#include "pipeline/comment_generator.h"
#include "conv_functions/conv2d.h"

using namespace torch;
namespace F = nn::functional;

typedef float conv_t;

typedef conv_runtime::ImageT<conv_t> (*GeneratedFunction) (conv_t* a, conv_t* b);

int default_padding[] = {0, 0};
int default_stride[] = {1, 1};
int default_dilation[] = {1, 1};
int default_groups = 1;
int default_batch_sz = 1;
int default_in_channels = 1;
int default_out_channels = 1;

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
            float threshold = 0.01 * expected_arr[i*w+j];
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

void test_conv2d(int iw, int ih, int ww, int wh, int batch_size, int in_channels, int out_channels, conv_runtime::ConvOptions conv_options, F::ConvFuncOptions<2> torch_options, string test_name, string test_details) {
    int64_t lo = 0;
    int64_t hi = 100;
    // generate image and kernel
    Tensor torch_input = torch::randint(lo, hi, {batch_size, in_channels, ih, iw}).to(torch::kFloat32); // using float here because dilation doesn't work with int
    Tensor torch_weight = torch::randint(lo, hi, {out_channels, in_channels/conv_options.groups, wh, ww}).to(torch::kFloat32);
    // std::cout << torch_input << std::endl;
    // std::cout << torch_weight << std::endl;
    // // get expected output
    Tensor torch_output = F::conv2d(torch_input, torch_weight, torch_options);
    // std::cout << torch_output << std::endl;
    // get actual output
    // Tensor torch_inp = torch_input.to(torch::kInt32);
    // Tensor torch_kernel = torch_weight.to(torch::kInt32);
    conv_runtime::ImageT<conv_t> conv_input = {.batch_size = batch_size, .in_channels = in_channels, .width = iw, .height = ih, .data = torch_input.data_ptr<conv_t>()};
    conv_runtime::KernelT<conv_t> conv_weight = {.out_channels = out_channels, .in_channels = in_channels, .width = ww, .height = wh, .data = torch_weight.data_ptr<conv_t>()};
    conv_runtime::ImageT<conv_t> conv_output = buildit_conv2d(conv_input, conv_weight, conv_options);
    // conv_output.print();
    compare(torch_output, conv_output, test_name, test_details);
}

// testing specialized conv2d
void test_static_conv2d(TestOptions opt, GeneratedFunction func, string test_name) {
    int64_t lo = 0;
    int64_t hi = 100;
    Tensor torch_input = torch::randint(lo, hi, {opt.batch_size, opt.in_channels, opt.ih, opt.iw}).to(torch::kFloat32); // using float here because dilation doesn't work with int in torch
    Tensor torch_weight = torch::randint(lo, hi, {opt.out_channels, opt.in_channels/default_groups, opt.wh, opt.ww}).to(torch::kFloat32);
    // covert options to torch arrays
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_stride = convert_to_expanding_array(opt.stride);
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(opt.dilation);
    torch_options = torch_options.stride(torch_stride).dilation(torch_dilation);
    if (opt.padding_same == 1) {
        torch_options = torch_options.padding(torch::kSame);
    } else {
        ExpandingArray<2> torch_padding = convert_to_expanding_array(opt.padding);
        torch_options = torch_options.padding(torch_padding);
    }
    // this is expected output
    Tensor torch_output = F::conv2d(torch_input, torch_weight, torch_options);
    // std::cout << torch_output << std::endl;
    // get actual output
    // Tensor torch_inp = torch_input.to(torch::kInt32);
    // Tensor torch_kernel = torch_weight.to(torch::kInt32);
    conv_t* inp_data = torch_input.data_ptr<conv_t>();
    conv_t* kernel_data = torch_weight.data_ptr<conv_t>();
    conv_runtime::ImageT<conv_t> conv_output = func(inp_data, kernel_data);
    // conv_output.print();
    // std::cout << conv_output.mult_cnt << std::endl;
    compare(torch_output, conv_output, test_name, "");
}



// unit tests

void test_default_options(int batch_sz, int in_channels, int out_channels) {
    conv_runtime::ConvOptions conv_options = {.stride = default_stride, .padding = default_padding, .dilation = default_dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    int iw[] = {5, 5, 10, 1};
    int ih[] = {5, 2, 10, 6};
    int ww[] = {3, 2, 6, 1};
    int wh[] = {3, 2, 3, 2};
    int n_tests = 4;
    string details[] = {"nxn image, nxn kernel", "nxn kernel", "nxn image", "1xn image"};
    for (int i = 0; i < n_tests; i++) {
        test_conv2d(iw[i], ih[i], ww[i], wh[i], batch_sz, in_channels, out_channels, conv_options, torch_options, "default_options", details[i]);
    }
}

void test_stride(int batch_sz, int in_channels, int out_channels) {
    int stride[2] = {2, 1};
    conv_runtime::ConvOptions conv_options = {.stride = stride, .padding = default_padding, .dilation = default_dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_stride = convert_to_expanding_array(stride);
    torch_options = torch_options.stride(torch_stride);
    int iw = 10;
    int ih = 8;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, in_channels, out_channels, conv_options, torch_options, "stride", "");
}

void test_dilation(int batch_sz, int in_channels, int out_channels) {
    int dilation[2] = {3, 2};
    conv_runtime::ConvOptions conv_options = {.stride = default_stride, .padding = default_padding, .dilation = dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    torch_options = torch_options.dilation(torch_dilation);
    int iw = 15;
    int ih = 20;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, in_channels, out_channels, conv_options, torch_options, "dilation", "");
}

void test_stride_dilation(int batch_sz, int in_channels, int out_channels) {
    int dilation[2] = {3, 2};
    int stride[2] = {2, 3};
    conv_runtime::ConvOptions conv_options = {.stride = stride, .padding = default_padding, .dilation = dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    ExpandingArray<2> torch_stride = convert_to_expanding_array(stride);
    torch_options = torch_options.dilation(torch_dilation).stride(torch_stride);
    int iw = 15;
    int ih = 20;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, in_channels, out_channels, conv_options, torch_options, "stride and dilation", "");
}

void test_padding_arr(int batch_sz, int in_channels, int out_channels) {
    int pad_arr[2] = {1, 2};
    conv_runtime::PaddingT padding = conv_runtime::PaddingT(pad_arr);
    conv_runtime::ConvOptions conv_options = {.stride = default_stride, .padding = padding, .dilation = default_dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_padding = convert_to_expanding_array(pad_arr);
    torch_options = torch_options.padding(torch_padding);
    int iw = 5;
    int ih = 5;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, in_channels, out_channels, conv_options, torch_options, "padding", "arr");
}

void test_padding_same(int batch_sz, int in_channels, int out_channels) {
    char pad_type[] = "same";
    conv_runtime::PaddingT padding = conv_runtime::PaddingT(pad_type);
    conv_runtime::ConvOptions conv_options = {.stride = default_stride, .padding = padding, .dilation = default_dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    torch_options = torch_options.padding(torch::kSame);
    int iw = 5;
    int ih = 5;
    int ww = 3;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, in_channels, out_channels, conv_options, torch_options, "padding", "same");
}

void test_stride_dilation_padding(int batch_sz, int in_channels, int out_channels) {
    int dilation[2] = {3, 2};
    int stride[2] = {2, 3};
    int pad_arr[2] = {3, 4};
    conv_runtime::PaddingT padding = conv_runtime::PaddingT(pad_arr);
    conv_runtime::ConvOptions conv_options = {.stride = stride, .padding = padding, .dilation = dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    ExpandingArray<2> torch_stride = convert_to_expanding_array(stride);
    ExpandingArray<2> torch_padding = convert_to_expanding_array(pad_arr);
    torch_options = torch_options.dilation(torch_dilation).stride(torch_stride).padding(torch_padding);
    int iw = 15;
    int ih = 20;
    int ww = 2;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, in_channels, out_channels, conv_options, torch_options, "stride, dilation, padding", "");
}

void test_dilation_padding_same(int batch_sz, int in_channels, int out_channels) {
    int dilation[2] = {3, 2};
    char pad_type[] = "same";
    conv_runtime::PaddingT padding = conv_runtime::PaddingT(pad_type);
    conv_runtime::ConvOptions conv_options = {.stride = default_stride, .padding = padding, .dilation = dilation, .groups = default_groups};
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();
    ExpandingArray<2> torch_dilation = convert_to_expanding_array(dilation);
    torch_options = torch_options.dilation(torch_dilation).padding(torch::kSame);
    int iw = 15;
    int ih = 20;
    int ww = 3;
    int wh = 3;
    test_conv2d(iw, ih, ww, wh, batch_sz, in_channels, out_channels, conv_options, torch_options, "dilation, padding", "same");
}

void test_batching_channels(int batch_sz, int in_ch, int out_ch) {
    std::cout << "Testing batch_size=" << batch_sz << ", in_channels=" << in_ch << ", out_channels=" << out_ch << std::endl;
    test_default_options(batch_sz, in_ch, out_ch);
    test_stride(batch_sz, in_ch, out_ch);
    test_dilation(batch_sz, in_ch, out_ch);
    test_stride_dilation(batch_sz, in_ch, out_ch);
    test_padding_arr(batch_sz, in_ch, out_ch);
    test_padding_same(batch_sz, in_ch, out_ch);
    test_stride_dilation_padding(batch_sz, in_ch, out_ch);
    test_dilation_padding_same(batch_sz, in_ch, out_ch);
    std::cout << "Done" << std::endl;
}

void compile_and_run(TestOptions opt, std::string test_name) {
    std::string flags = "";
    auto fptr = (GeneratedFunction)pipeline::commented_code_generator::compile_function(
            static_conv2d_with_tiled_loops, flags, opt.iw, opt.ih, opt.ww, opt.wh, opt.batch_size, opt.in_channels, 
            opt.out_channels, opt.stride, opt.dilation, opt.padding, opt.padding_same
            );
    test_static_conv2d(opt, fptr, test_name);
}

void test_static_all() {

    std::cout << "testing specialized code" << std::endl;
    TestOptions options;
    int stride[2];
    int dilation[2];
    int padding[2];
    int padding_same;

    // default
    options = {.iw = 5, .ih = 5, .ww = 3, .wh = 3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    compile_and_run(options, "default_im5x5_w3x3");

    // stride
    stride[0] = 2;
    stride[1] = 1;
    options = {.iw = 10, .ih = 8, .ww = 2, .wh = 3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    compile_and_run(options, "stride2x1_im8x10_w3x2");

    // dilation
    dilation[0] = 3;
    dilation[1] = 2;
    options = {.iw = 15, .ih = 20, .ww = 2, .wh = 3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = dilation};
    compile_and_run(options, "dil3x2_im20x15_w3x2");

    // stride and dilation
    dilation[0] = 3;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 3;
    options = {.iw = 15, .ih = 20, .ww = 2, .wh = 3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = default_padding, .dilation = dilation};
    compile_and_run(options, "stride2x3_dil3x2_im20x15_w3x2");

    // padding arr
    padding[0] = 1;
    padding[1] = 2;
    options = {.iw = 5, .ih = 5, .ww = 2, .wh = 3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = padding, .dilation = default_dilation};
    compile_and_run(options, "pad1x2_im5x5_w3x2");
    // padding same
    options = {.iw = 5, .ih = 5, .ww = 3, .wh = 3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 1, .padding = default_padding, .dilation = default_dilation};
    compile_and_run(options, "padsame_im5x5_w3x3");
    // stride, dilation, padding arr
    dilation[0] = 3;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 3;
    padding[0] = 3;
    padding[1] = 4;
    options = {.iw = 15, .ih = 20, .ww = 2, .wh = 3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    compile_and_run(options, "dil3x2_stride2x3_pad3x4_im15x20_w3x2");

    // dilation, padding same
    dilation[0] = 3;
    dilation[1] = 2;
    options = {.iw = 15, .ih = 20, .ww = 3, .wh = 3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 1, .padding = default_padding, .dilation = dilation};
    compile_and_run(options, "dil3x2_padsame_im15x20_w3x3");

    // batching, stride, dilation, padding
    dilation[0] = 2;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 4;
    padding[0] = 5;
    padding[1] = 4;
    options = {.iw = 20, .ih = 20, .ww = 3, .wh = 3, .batch_size = 5, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    compile_and_run(options, "dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5");

    // batching, stride, dilation, padding, in channels, out channels
    dilation[0] = 2;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 4;
    padding[0] = 5;
    padding[1] = 4;
    options = {.iw = 20, .ih = 20, .ww = 5, .wh = 5, .batch_size = 4, .in_channels = 3, .out_channels = 5, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    compile_and_run(options, "dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5");

    // regression tests from timing
    options = {.iw = 100, .ih = 100, .ww = 10, .wh = 10, .batch_size = 10, .in_channels = 10, .out_channels = 10, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    compile_and_run(options, "im100x100_w10x10_batch10_inch10_outch10");

    stride[0] = 4;
    stride[1] = 4;
    options = {.iw = 100, .ih = 100, .ww = 10, .wh = 10, .batch_size = 10, .in_channels = 5, .out_channels = 10, 
                                    .stride = stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    compile_and_run(options, "stride4x4_im100x100_w10x10_batch10_inch10_outch10");

    options = {.iw = 10, .ih = 10, .ww = 5, .wh = 5, .batch_size = 10, .in_channels = 5, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    compile_and_run(options, "im10x10_w5x5_batch10_inch5_outch1");

    dilation[0] = 4;
    dilation[1] = 4;
    padding[0] = 10;
    padding[1] = 10;
    options = {.iw = 10, .ih = 10, .ww = 3, .wh = 3, .batch_size = 1, .in_channels = 5, .out_channels = 3, 
                                    .stride = default_stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    compile_and_run(options, "pad10x10_dil4x4_im10x10_w3x3_batch1_inch5_outch3");
}


int main() {
    // test_default_options(default_batch_sz, default_in_channels, default_out_channels);
    // test_stride(default_batch_sz, default_in_channels, default_out_channels);
    // test_dilation(default_batch_sz, default_in_channels, default_out_channels);
    // test_stride_dilation(default_batch_sz, default_in_channels, default_out_channels);
    // test_padding_arr(default_batch_sz, default_in_channels, default_out_channels);
    // test_padding_same(default_batch_sz, default_in_channels, default_out_channels);
    // test_stride_dilation_padding(default_batch_sz, default_in_channels, default_out_channels);
    // test_dilation_padding_same(default_batch_sz, default_in_channels, default_out_channels);
    // test_batching_channels(4, default_in_channels, default_out_channels); // batching
    // test_batching_channels(default_batch_sz, 4, default_out_channels); // in channels
    // test_batching_channels(2, 4, default_out_channels); // both batches and in channels
    // test_batching_channels(default_batch_sz, default_in_channels, 3); // out_channels
    // test_batching_channels(default_in_channels, 3, 4); // both in and out channels
    // test_batching_channels(5, default_in_channels, 2); // batching and out channels
    // test_batching_channels(3, 4, 5); // test all

    test_static_all();

}