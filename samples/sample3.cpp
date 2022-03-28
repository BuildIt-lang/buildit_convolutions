#include <fstream>
#include "blocks/c_code_generator.h"
#include "builder/dyn_var.h"
#include "blocks/rce.h"
#include "conv_functions/conv2d.h"
#include "pipeline/conv.h"
#include "pipeline/comment_generator.h"

using builder::dyn_var;
using builder::static_var;

int main() {
    std::ofstream code_file;
    code_file.open("./generated_code/specialized_timing_code.h");
    code_file << "#include <assert.h>\n" << std::endl;
    
    int n_runs = 8;
    int iw[] = {10, 100, 100, 100, 100, 1000, 1000, 1000};
    int ih[] = {10, 100, 100, 100, 100, 1000, 1000, 1000};
    int kw[] = {3, 10, 10, 10, 10, 10, 10, 10};
    int kh[] = {3, 10, 10, 10, 10, 10, 10, 10};
    int batch_size[] = {10, 10, 10, 10, 10, 10, 10, 10};
    int in_channels[] = {5, 5, 5, 10, 10, 10, 10, 100};
    int out_channels[] = {10, 10, 10, 10, 1, 10, 100, 10};
    int stride[][2] = {{1, 1}, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    int padding[][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
    int dilation[][2] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    int padding_same[] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::string func_names[] = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"};
    for (int i = 0; i < n_runs; i ++) {
        auto ast = builder::builder_context().extract_function_ast(static_conv2d, func_names[i], iw[i], ih[i], kw[i], kh[i], batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i]);
        block::eliminate_redundant_vars(ast);
        pipeline::commented_code_generator::generate_code(ast, code_file, 0);
        code_file << "\n" << std::endl;
    }
    code_file.close();
	return 0;
}