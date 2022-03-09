#include <fstream>
#include "blocks/c_code_generator.h"
#include "builder/static_var.h"
#include "builder/dyn_var.h"
#include <iostream>
#include "conv_functions/conv2d.h"
#include "conv_functions/runtime.h"
#include "blocks/rce.h"
#include "pipeline/conv.h"


using builder::dyn_var;
using builder::static_var;


static void run_conv2d() {
    ImageT input;
    input.width = 5;
    input.height = 3;
    input.data = conv::runtime::conv_malloc((int)sizeof(int) * input.width * input.height);

    KernelT weight;
    weight.width = 2;
    weight.height = 2;
    weight.data = conv::runtime::conv_malloc((int)sizeof(int) * weight.width * weight.height);


    // generate input
    for (dyn_var<int> i = 0; i < input.height; i = i + 1) {
        for (dyn_var<int> j = 0; j < input.width; j = j + 1) {
            input.data[i * input.width + j] = i * input.width + j;
        }
    }

    // generate kernel
    for (dyn_var<int> i = 0; i < weight.height; i = i + 1) {
        for (dyn_var<int> j = 0; j < weight.width; j = j + 1) {
            weight.data[i * weight.width + j] = i * weight.width + j;
        }
    }
    

    // TensorT output = conv2d(input, weight, options);

    // output.print();

    input.print();
    weight.print();

    conv::runtime::conv_free(input.data);
    conv::runtime::conv_free(weight.data);
    // conv::runtime::conv_free(output.data);
}


int main() {

    std::ofstream code_file;
    code_file.open("./generated_code/sample2.cpp");
    auto ast = builder::builder_context().extract_function_ast(run_conv2d, "run_conv2d");
    pipeline::generate_conv_code(ast, code_file, "run_conv2d");
    code_file.close();
	block::c_code_generator::generate_code(ast, std::cout, 0);
	return 0;
}