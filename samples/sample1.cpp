#include <fstream>

#include "conv_functions/conv2d.h"
#include "conv_functions/runtime.h"
#include "blocks/c_code_generator.h"
#include "blocks/rce.h"
#include "builder/static_var.h"
#include "builder/dyn_var.h"
#include "pipeline/conv.h"


using builder::dyn_var;
using builder::static_var;

static void run_conv2d() {
    dyn_var<int> input_size = 10;
    dyn_var<int> weight_size = 5;
    dyn_var<int> output_size = input_size - weight_size + 1;

    dyn_var<int*> input = conv::runtime::conv_malloc((int)sizeof(int) * input_size * input_size);
    dyn_var<int*> weight = conv::runtime::conv_malloc((int)sizeof(int) * weight_size * weight_size);
    dyn_var<int*> output = conv::runtime::conv_malloc((int)sizeof(int) * output_size * output_size);

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

    static_var<int> n_iters = 100;

    conv::runtime::start_timer();
    for (dyn_var<int> i = 0; i < n_iters; i = i + 1) {
        conv2d_nxn(input, weight, output, input_size, weight_size, output_size);
    }
    dyn_var<float> t = conv::runtime::stop_timer() / n_iters;
    conv::runtime::print_time(t);

    conv::runtime::print_matrix(output, output_size);

    conv::runtime::conv_free(input);
    conv::runtime::conv_free(weight);
    conv::runtime::conv_free(output);
}

int main(int argc, char* argv[]) {
    std::ofstream code_file;
    code_file.open("./generated_code/sample1.cpp");
    auto ast = builder::builder_context().extract_function_ast(run_conv2d, "run_conv2d");
    pipeline::generate_conv_code(ast, code_file, "run_conv2d");
    code_file.close();
	block::c_code_generator::generate_code(ast, std::cout, 0);
	return 0;
}
