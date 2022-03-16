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
    code_file.open("./generated_code/specialized_test_code.h");
    // code_file << "#include \"runtime_functions.h\"" << std::endl;
    // code_file << "#include \"runtime_types.h\"\n" << std::endl;
    code_file << "#include <assert.h>\n" << std::endl;
    
    int num_tests = 1;
    std::string func_name[] = {"conv2d_default_im5x5_w3x3"};
    int stride[][2] = {{1, 1},};
    int dilation[][2] = {{1, 1}};
    int padding[][2] ={{0, 0}};
    int padding_same[] = {0};
    // int iw[] = {5};
    // int ih[] = {5};
    // int ww[] = {3};
    // int wh[] = {3};
    // int batch_size[] = {1};
    // int in_channels[] = {1};
    // int out_channels[] = {1};
    for (int i = 0; i < num_tests; i ++) {
        auto ast = builder::builder_context().extract_function_ast(static_conv2d, func_name[i], stride[i], dilation[i], padding[i], padding_same[i]);
        block::eliminate_redundant_vars(ast);
        pipeline::commented_code_generator::generate_code(ast, code_file, 0);
        code_file << "\n" << std::endl;
    }
    code_file.close();
	return 0;
}