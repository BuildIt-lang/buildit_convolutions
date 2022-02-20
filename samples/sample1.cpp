#include "conv_functions/conv2d.h"
#include "conv_functions/runtime.h"
#include "blocks/c_code_generator.h"
#include "builder/static_var.h"
#include "builder/dyn_var.h"

using builder::dyn_var;
using builder::static_var;

static void run_conv2d() {
    dyn_var<int> input_size = 6;
    dyn_var<int> weight_size = 3;
    dyn_var<int> output_size = input_size - weight_size + 1;

    dyn_var<int*> input = conv::runtime::malloc((int)sizeof(int) * input_size * input_size);
    dyn_var<int*> weight = conv::runtime::malloc((int)sizeof(int) * weight_size * weight_size);
    dyn_var<int*> output = conv::runtime::malloc((int)sizeof(int) * output_size * output_size);

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
    conv2d(input, weight, output, input_size, weight_size, output_size);

    conv::runtime::free(input);
    conv::runtime::free(weight);
    conv::runtime::free(output);
}

int main(int argc, char* argv[]) {
	block::c_code_generator::generate_code(builder::builder_context().extract_function_ast(run_conv2d, "run_conv2d"), std::cout, 0);
	return 0;
}
