#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"
#include "conv_functions/runtime.h"
#include "blocks/c_code_generator.h"
#include "conv_functions/conv2d.h"

using builder::dyn_var;
using conv::TensorT;
using conv::ConvOptions;

TensorT conv2d(TensorT input, TensorT weight, ConvOptions opt) {

    TensorT output;
    output.height = (input.height - opt.dilation[0] * (weight.height - 1) - 1) / opt.stride[0] + 1;
    output.width = (input.width - opt.dilation[1] * (weight.width - 1) - 1) / opt.stride[1] + 1;
    dyn_var<int> size = output.width * output.height;
    output.data = conv::runtime::conv_malloc((int)sizeof(int)*size);
    dyn_var<int> idx;
    for (dyn_var<int> h = 0; h < output.height; h = h + 1) {
        for (dyn_var<int> w = 0; w < output.width; w = w + 1) {
            idx =  h * output.width + w;
            output.data[idx] = 0;
            for (dyn_var<int> i = 0; i < weight.height; i = i + 1){
                for (dyn_var<int> j = 0; j < weight.width; j = j + 1) {
                    output.data[idx] = output.data[idx] +
                    input.data[(h * opt.stride[0] + i * opt.dilation[0]) * input.width + (w * opt.stride[1] + j * opt.dilation[1])] * weight.data[i * weight.width + j];
                }
            }
        }
    }
    return output;
}

// convolution for NxN matrices
void conv2d_nxn(dyn_var<int*> input, dyn_var<int*> weight, dyn_var<int*> output, dyn_var<int> input_size, dyn_var<int> weight_size, dyn_var<int> output_size) {
    for (dyn_var<int> i = 0; i < output_size; i = i + 1) {
        for (dyn_var<int> j = 0; j < output_size; j = j + 1) {
            output[i*output_size+j] = 0;
            for (dyn_var<int> ki = 0; ki < weight_size; ki = ki + 1) {
                for (dyn_var<int> kj = 0; kj < weight_size; kj = kj + 1) {
                    output[i*output_size+j] = output[i*output_size+j] + weight[ki*weight_size+kj] * input[(i+ki)*input_size+(j+kj)];
                }
            }
        }
    }
}
