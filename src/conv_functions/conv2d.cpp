#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"
#include "conv_functions/runtime.h"
#include "blocks/c_code_generator.h"
#include "conv_functions/conv2d.h"

using builder::dyn_var;
using conv::TensorT;

TensorT conv2d(TensorT input, TensorT weight, int stride, int padding, int dilation, int groups) {

    TensorT output;
    output.width = input.width - weight.width + 1;
    output.height = input.height - weight.height + 1;
    dyn_var<int> size = output.width * output.height;
    output.data = conv::runtime::conv_malloc((int)sizeof(int)*size);
    
    for (dyn_var<int> h = 0; h < output.height; h = h + 1) {
        for (dyn_var<int> w = 0; w < output.width; w = w + 1) {
            output.data[h * output.width + w] = 0;
            for (dyn_var<int> i = 0; i < weight.height; i = i + 1){
                for (dyn_var<int> j = 0; j < weight.width; j = j + 1) {
                    output.data[h * output.width + w] = output.data[h * output.width + w] +
                    input.data[(h+i) * input.width + (w+j)] * weight.data[i * weight.width + j];
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
