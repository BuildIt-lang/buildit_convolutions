#include <iostream>
#include <torch/torch.h>
#include "test_types.h"
#include <assert.h>
#include "runtime_functions.h"
#include "runtime_types.h"
#include "pipeline/conv_code_generator.h"
#include "conv_functions/conv2d.h"
#include "test_conv2d_code.h"
#include "test_convnd_code.h"

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

void compare_dim(conv_t* result, conv_t* expected, int* bounds , int dim, int idx, int depth) {
    if (dim == 0) {
        float diff = std::abs(result[idx] - expected[idx]);
        float threshold = 0.01 * expected[idx];
        if (diff > threshold) {
            std::cout << "expected: " << expected[idx] << ", got: " << result[idx] << std::endl;
        }
        assert(diff <= threshold);
    } else {
        for (int i = 0; i < bounds[dim - 1]; i++) {
            compare_dim(result, expected, bounds, dim - 1, idx + i * depth, depth * bounds[dim - 1]);
        }
    }
}

void compare(Tensor expected, conv_runtime::ImageT<conv_t> result, string test_name, string test_details, int ndims) {
    std::cout << "Running test: " << test_name << " " << test_details;
    assert(result.batch_size == expected.size(0));
    assert(result.in_channels == expected.size(1));
    for (int d = 0; d < ndims; d++) {
        assert (result.dims[d] == expected.size(2 + d));
    }
    conv_t* expected_arr = expected.data_ptr<conv_t>();
    compare_dim(result.data, expected_arr, result.dims, ndims, 0, 1);
    std::cout << ": PASSED" << std::endl;
}

template <unsigned int d>
ExpandingArray<d> convert_to_expanding_array(int* arr) {
    int64_t arr_long[d];
    for (int i = 0; i < d; i++) {
        arr_long[i] = (int64_t)arr[i];
    }
    ArrayRef<int64_t> arr_ref = ArrayRef<int64_t>(arr_long, d);
    ExpandingArray<d> expanding_arr = ExpandingArray<d>(arr_ref);
    return expanding_arr;
}


// testing specialized conv2d
void test_conv2d(TestOptions opt, GeneratedFunction func, string test_name) {
    int64_t lo = 0;
    int64_t hi = 100;
    Tensor torch_input = torch::randint(lo, hi, {opt.batch_size, opt.in_channels, opt.img[0], opt.img[1]}).to(torch::kFloat32); // using float here because dilation doesn't work with int in torch
    Tensor torch_weight = torch::randint(lo, hi, {opt.out_channels, opt.in_channels/default_groups, opt.ker[0], opt.ker[1]}).to(torch::kFloat32);
    // covert options to torch arrays
    F::ConvFuncOptions<2> torch_options = F::Conv2dFuncOptions();; 
    ExpandingArray<2> torch_stride = convert_to_expanding_array<2>(opt.stride);
    ExpandingArray<2> torch_dilation = convert_to_expanding_array<2>(opt.dilation);
    torch_options = torch_options.stride(torch_stride).dilation(torch_dilation);
    if (opt.padding_same == 1) {
        torch_options = torch_options.padding(torch::kSame);
    } else {
        ExpandingArray<2> torch_padding = convert_to_expanding_array<2>(opt.padding);
        torch_options = torch_options.padding(torch_padding);
    }
    // this is expected output
    Tensor torch_output = F::conv2d(torch_input, torch_weight, torch_options);

    // std::cout << torch_output << std::endl;
    // get actual output
    conv_t* inp_data = torch_input.data_ptr<conv_t>();
    conv_t* kernel_data = torch_weight.data_ptr<conv_t>();
    conv_runtime::ImageT<conv_t> conv_output = func(inp_data, kernel_data);
    // conv_output.print();
    // std::cout << conv_output.mult_cnt << std::endl;
    compare(torch_output, conv_output, test_name, "", 2);
}


void test_conv3d(TestOptions opt, GeneratedFunction func, string test_name) {
    int64_t lo = 0;
    int64_t hi = 100;
    Tensor torch_input = torch::randint(lo, hi, {opt.batch_size, opt.in_channels, opt.img[0], opt.img[1], opt.img[2]}).to(torch::kFloat32); // using float here because dilation doesn't work with int in torch
    Tensor torch_weight = torch::randint(lo, hi, {opt.out_channels, opt.in_channels/default_groups, opt.ker[0], opt.ker[1], opt.ker[2]}).to(torch::kFloat32);
    // covert options to torch arrays
    F::ConvFuncOptions<3> torch_options = F::Conv3dFuncOptions();; 
    ExpandingArray<3> torch_stride = convert_to_expanding_array<3>(opt.stride);
    ExpandingArray<3> torch_dilation = convert_to_expanding_array<3>(opt.dilation);
    torch_options = torch_options.stride(torch_stride).dilation(torch_dilation);
    if (opt.padding_same == 1) {
        torch_options = torch_options.padding(torch::kSame);
    } else {
        ExpandingArray<3> torch_padding = convert_to_expanding_array<3>(opt.padding);
        torch_options = torch_options.padding(torch_padding);
    }
    // this is expected output
    Tensor torch_output = F::conv3d(torch_input, torch_weight, torch_options);

    // std::cout << torch_output << std::endl;
    // get actual output
    conv_t* inp_data = torch_input.data_ptr<conv_t>();
    conv_t* kernel_data = torch_weight.data_ptr<conv_t>();
    conv_runtime::ImageT<conv_t> conv_output = func(inp_data, kernel_data);
    // conv_output.print();
    // std::cout << conv_output.mult_cnt << std::endl;
    compare(torch_output, conv_output, test_name, "", 3);
}

void compile_and_run(TestOptions opt, std::string test_name, int ndims) {
    std::string flags = "";
    auto fptr = (GeneratedFunction)pipeline::conv_code_generator::compile_function(
            static_conv2d_with_tiled_loops, flags, opt.img[1], opt.img[0], opt.ker[1], opt.ker[0], opt.batch_size, opt.in_channels, 
            opt.out_channels, opt.stride, opt.dilation, opt.padding, opt.padding_same
            );
    if (ndims == 2) test_conv2d(opt, fptr, test_name);
    else if (ndims == 3) test_conv3d(opt, fptr, test_name);
}

void test_conv2d() {

    std::cout << "testing 2d" << std::endl;
    TestOptions options;
    int stride[2];
    int dilation[2];
    int padding[2];
    int padding_same;
    int ndims = 2;

    // default
    int img1[2] = {5, 5};
    int ker1[2] = {3, 3};
    options = {.img = img1, .ker = ker1, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    // compile_and_run(options, "default_im5x5_w3x3");
    test_conv2d(options, &conv2d_default_im5x5_w3x3, "default_im5x5_w3x3");

    int img2[2] = {2, 5};
    int ker2[2] = {2, 2};
    options = {.img = img2, .ker = ker2, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    test_conv2d(options, &conv2d_default_im5x2_w2x2, "default_im5x2_w2x2");

    int img3[2] = {10, 10};
    int ker3[2] = {3, 6};
    options = {.img = img3, .ker = ker3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    test_conv2d(options, &conv2d_default_im10x10_w6x3, "default_im10x10_w6x3");

    int img4[2] = {6, 1};
    int ker4[2] = {2, 1};
    options = {.img = img4, .ker = ker4, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    test_conv2d(options, &conv2d_default_im1x6_w1x2, "default_im1x6_w1x2");
    
    // stride
    stride[0] = 2;
    stride[1] = 1;
    int img5[2] = {8, 10};
    int ker5[2] = {3, 2};
    options = {.img = img5, .ker = ker5, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    // compile_and_run(options, "stride2x1_im8x10_w3x2");
    test_conv2d(options, &conv2d_stride2x1_im8x10_w3x2, "stride2x1_im8x10_w3x2");

    // dilation
    dilation[0] = 3;
    dilation[1] = 2;
    int img6[2] = {20, 15};
    int ker6[2] = {3, 2};
    options = {.img = img6, .ker = ker6, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = dilation};
    // compile_and_run(options, "dil3x2_im20x15_w3x2");
    test_conv2d(options, &conv2d_dil3x2_im20x15_w3x2, "dil3x2_im20x15_w3x2");

    // stride and dilation
    dilation[0] = 3;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 3;
    int img7[2] = {20, 15};
    int ker7[2] = {3, 2};
    options = {.img = img7, .ker = ker7, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = default_padding, .dilation = dilation};
    // compile_and_run(options, "stride2x3_dil3x2_im20x15_w3x2");
    test_conv2d(options, &conv2d_stride2x3_dil3x2_im20x15_w3x2, "stride2x3_dil3x2_im20x15_w3x2");

    // padding arr
    padding[0] = 1;
    padding[1] = 2;
    int img8[2] = {5, 5};
    int ker8[2] = {3, 2};
    options = {.img = img8, .ker = ker8, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = padding, .dilation = default_dilation};
    // compile_and_run(options, "pad1x2_im5x5_w3x2");
    test_conv2d(options, &conv2d_pad1x2_im5x5_w3x2, "pad1x2_im5x5_w3x2");

    // padding same
    int img9[2] = {5, 5};
    int ker9[2] = {3, 3};
    options = {.img = img9, .ker = ker9, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 1, .padding = default_padding, .dilation = default_dilation};
    // compile_and_run(options, "padsame_im5x5_w3x3");
    test_conv2d(options, &conv2d_padsame_im5x5_w3x3, "padsame_im5x5_w3x3");

    // stride, dilation, padding arr
    dilation[0] = 3;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 3;
    padding[0] = 3;
    padding[1] = 4;
    int img10[2] = {20, 15};
    int ker10[2] = {3, 2};
    options = {.img = img10, .ker = ker10, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    // compile_and_run(options, "dil3x2_stride2x3_pad3x4_im15x20_w3x2");
    test_conv2d(options, &conv2d_dil3x2_stride2x3_pad3x4_im15x20_w3x2, "dil3x2_stride2x3_pad3x4_im15x20_w3x2");

    // dilation, padding same
    dilation[0] = 3;
    dilation[1] = 2;
    int img11[2] = {20, 15};
    int ker11[2] = {3, 3};
    options = {.img = img11, .ker = ker11, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 1, .padding = default_padding, .dilation = dilation};
    // compile_and_run(options, "dil3x2_padsame_im15x20_w3x3");
    test_conv2d(options, &conv2d_dil3x2_padsame_im15x20_w3x3, "dil3x2_padsame_im15x20_w3x3");

    // batching, stride, dilation, padding
    dilation[0] = 2;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 4;
    padding[0] = 5;
    padding[1] = 4;
    int img12[2] = {20, 20};
    int ker12[2] = {3, 3};
    options = {.img = img12, .ker = ker12, .batch_size = 5, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    // compile_and_run(options, "dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5");
    test_conv2d(options, &conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5, "dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5");

    // in channels, stride, dilation, padding
    dilation[0] = 2;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 4;
    padding[0] = 5;
    padding[1] = 4;
    int img13[2] = {20, 20};
    int ker13[2] = {3, 3};
    options = {.img = img13, .ker = ker13, .batch_size = 1, .in_channels = 5, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    test_conv2d(options, &conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_inch5, "dil2x2_stride2x4_pad5x4_im20x20_w3x3_inch5");

    // out channels, stride, dilation, padding
    dilation[0] = 2;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 4;
    padding[0] = 5;
    padding[1] = 4;
    int img14[2] = {20, 20};
    int ker14[2] = {3, 3};
    options = {.img = img14, .ker = ker14, .batch_size = 1, .in_channels = 1, .out_channels = 6, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    test_conv2d(options, &conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_outch6, "dil2x2_stride2x4_pad5x4_im20x20_w3x3_outch6");

    // batching, stride, dilation, padding, in channels, out channels
    dilation[0] = 2;
    dilation[1] = 2;
    stride[0] = 2;
    stride[1] = 4;
    padding[0] = 5;
    padding[1] = 4;
    int img15[2] = {20, 20};
    int ker15[2] = {5, 5};
    options = {.img = img15, .ker = ker15, .batch_size = 4, .in_channels = 3, .out_channels = 5, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    // compile_and_run(options, "dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5");
    test_conv2d(options, &conv2d_dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5, "dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5");

    // regression tests from timing
    int img16[2] = {100, 100};
    int ker16[2] = {10, 10};
    options = {.img = img16, .ker = ker16, .batch_size = 10, .in_channels = 10, .out_channels = 10, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    // compile_and_run(options, "im100x100_w10x10_batch10_inch10_outch10");
    test_conv2d(options, &conv2d_im100x100_w10x10_batch10_inch10_outch10, "im100x100_w10x10_batch10_inch10_outch10");

    stride[0] = 4;
    stride[1] = 4;
    int img17[2] = {100, 100};
    int ker17[2] = {10, 10};
    options = {.img = img17, .ker = ker17, .batch_size = 10, .in_channels = 5, .out_channels = 10, 
                                    .stride = stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    // compile_and_run(options, "stride4x4_im100x100_w10x10_batch10_inch5_outch10");
    test_conv2d(options, &conv2d_stride4x4_im100x100_w10x10_batch10_inch5_outch10, "stride4x4_im100x100_w10x10_batch10_inch5_outch10");

    int img18[2] = {10, 10};
    int ker18[2] = {5, 5};
    options = {.img = img18, .ker = ker18, .batch_size = 10, .in_channels = 5, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    // compile_and_run(options, "im10x10_w5x5_batch10_inch5_outch1");
    test_conv2d(options, &conv2d_im10x10_w5x5_batch10_inch5_outch1, "im10x10_w5x5_batch10_inch5_outch1");

    dilation[0] = 4;
    dilation[1] = 4;
    padding[0] = 10;
    padding[1] = 10;
    int img19[2] = {10, 10};
    int ker19[2] = {3, 3};
    options = {.img = img19, .ker = ker19, .batch_size = 1, .in_channels = 5, .out_channels = 3, 
                                    .stride = default_stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    // compile_and_run(options, "pad10x10_dil4x4_im10x10_w3x3_batch1_inch5_outch3");
    test_conv2d(options, &conv2d_pad10x10_dil4x4_im10x10_w3x3_batch1_inch5_outch3, "pad10x10_dil4x4_im10x10_w3x3_batch1_inch5_outch3");
    
}


void test_conv3d() {

    std::cout << "testing 3d" << std::endl;
    TestOptions options;
    int stride[3];
    int dilation[3];
    int padding[3];
    int padding_same;
    int ndims = 2;

    // default
    int img1[3] = {5, 5, 5};
    int ker1[3] = {3, 3, 3};
    options = {.img = img1, .ker = ker1, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    test_conv3d(options, &conv3d_default_img5x5x5_ker3x3x3, "default_img5x5x5_ker3x3x3");

    // default
    int img2[3] = {6, 4, 10};
    int ker2[3] = {2, 3, 2};
    options = {.img = img2, .ker = ker2, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    test_conv3d(options, &conv3d_default_img6x4x10_ker2x3x2, "default_img6x4x10_ker2x3x2");

    // stride
    stride[0] = 2;
    stride[1] = 3;
    stride[2] = 3;
    int img3[3] = {8, 12, 10};
    int ker3[3] = {2, 2, 2};
    options = {.img = img3, .ker = ker3, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = default_padding, .dilation = default_dilation};
    test_conv3d(options, &conv3d_str2x3x3_img8x12x10_ker2x2x2, "str2x3x3_img8x12x10_ker2x2x2");

    // dilation
    dilation[0] = 3;
    dilation[1] = 2;
    dilation[2] = 1;
    int img4[3] = {12, 10, 8};
    int ker4[3] = {2, 1, 2};
    options = {.img = img4, .ker = ker4, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = default_padding, .dilation = dilation};
    test_conv3d(options, &conv3d_dil3x2x1_img12x10x8_ker2x1x2, "dil3x2x1_img12x10x8_ker2x1x2");

    // padding
    padding[0] = 2;
    padding[1] = 4;
    padding[2] = 3;
    int img5[3] = {10, 10, 8};
    int ker5[3] = {4, 2, 3};
    options = {.img = img5, .ker = ker5, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 0, .padding = padding, .dilation = default_dilation};
    test_conv3d(options, &conv3d_pad2x4x3_img10x10x8_ker4x2x3, "pad2x4x3_img10x10x8_ker4x2x3");

    // padding same
    int img6[3] = {10, 9, 9};
    int ker6[3] = {3, 3, 3};
    options = {.img = img6, .ker = ker6, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = default_stride, .padding_same = 1, .padding = default_padding, .dilation = default_dilation};
    test_conv3d(options, &conv3d_padsame_img10x9x9_ker3x3x3, "padsame_img10x9x9_ker3x3x3");

    // stride padding dilation 
    padding[0] = 1;
    padding[1] = 2;
    padding[2] = 3;
    stride[0] = 2;
    stride[1] = 2;
    stride[2] = 2;
    dilation[0] = 2;
    dilation[1] = 3;
    dilation[2] = 2;
    int img7[3] = {15, 15, 10};
    int ker7[3] = {5, 3, 5};
    options = {.img = img7, .ker = ker7, .batch_size = 1, .in_channels = 1, .out_channels = 1, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    test_conv3d(options, &conv3d_str2x2x2_dil_2x3x2_pad_1x2x3_img_15x15x10_ker5x3x5, "str2x2x2_dil_2x3x2_pad_1x2x3_img_15x15x10_ker5x3x5");

    // all params
    padding[0] = 2;
    padding[1] = 2;
    padding[2] = 2;
    stride[0] = 1;
    stride[1] = 2;
    stride[2] = 1;
    dilation[0] = 2;
    dilation[1] = 1;
    dilation[2] = 2;
    int img8[3] = {10, 6, 8};
    int ker8[3] = {2, 2, 2};
    options = {.img = img8, .ker = ker8, .batch_size = 3, .in_channels = 4, .out_channels = 5, 
                                    .stride = stride, .padding_same = 0, .padding = padding, .dilation = dilation};
    test_conv3d(options, &conv3d_n3_ic4_oc5_str1x2x1_dil2x1x2_pad2x2x2_img10x6x8_ker2x2x2, "n3_ic4_oc5_str1x2x1_dil2x1x2_pad2x2x2_img10x6x8_ker2x2x2");

    
}

int main() {
    test_conv2d();
    test_conv3d();
}
