// Include the headers
#include "blocks/c_code_generator.h"
#include "builder/static_var.h"
#include "builder/dyn_var.h"
#include <iostream>
#include <stdio.h> 
#include <stdlib.h>

// Include the BuildIt types
using builder::dyn_var;
using builder::static_var;

namespace runtime {
    dyn_var<void*(int)> malloc("runtime::malloc");
    dyn_var<void(int*)>free("runtime::free");
}

void buildit_conv2d(dyn_var<int*> input, dyn_var<int*> weight, dyn_var<int*> output, dyn_var<int> input_size, dyn_var<int> weight_size, dyn_var<int> output_size);
void run_conv2d();

void buildit_conv2d(dyn_var<int*> input, dyn_var<int*> weight, dyn_var<int*> output, dyn_var<int> input_size, dyn_var<int> weight_size, dyn_var<int> output_size) {
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

void run_conv2d() {
    dyn_var<int> input_size = 6;
    dyn_var<int> weight_size = 3;
    dyn_var<int> output_size = input_size - weight_size + 1;

    dyn_var<int*> input = runtime::malloc((int)sizeof(int) * input_size * input_size);
    dyn_var<int*> weight = runtime::malloc((int)sizeof(int) * weight_size * weight_size);
    dyn_var<int*> output = runtime::malloc((int)sizeof(int) * output_size * output_size);

    // generate input
    for (dyn_var<int> i = 0; i < input_size; i = i + 1) {
        for (dyn_var<int> j = 0; j < input_size; j = j + 1) {
            input[i * input_size + j] = i * input_size + j;
        }
    }

    // generate kernel
    for (dyn_var<int> i = 0; i < weight_size; i = i + 1) {
        for (dyn_var<int> j = 0; j < weight_size; j = j + 1) {
            weight[i * weight_size + j] = i * weight_size + j;
        }
    }
    buildit_conv2d(input, weight, output, input_size, weight_size, output_size);

    runtime::free(input);
    runtime::free(weight);
    runtime::free(output);
}

int main(int argc, char* argv[]) {
	block::c_code_generator::generate_code(builder::builder_context().extract_function_ast(run_conv2d, "run_conv2d"), std::cout, 0);
	return 0;
}
