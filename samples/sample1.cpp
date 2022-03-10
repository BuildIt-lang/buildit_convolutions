#include <fstream>
#include "blocks/c_code_generator.h"
#include "builder/dyn_var.h"
#include "blocks/rce.h"
#include "conv_functions/conv2d.h"
#include "pipeline/conv.h"
#include "pipeline/comment_generator.h"

using builder::dyn_var;

int main() {
    std::ofstream code_file;
    code_file.open("./generated_code/buildit_conv2d.h");
    code_file << "#include \"runtime_functions.h\"" << std::endl;
    code_file << "#include \"runtime_types.h\"\n" << std::endl;
    code_file << "#include <assert.h>\n" << std::endl;
    auto ast = builder::builder_context().extract_function_ast(conv2d, "buildit_conv2d");
    block::eliminate_redundant_vars(ast);
	// block::c_code_generator::generate_code(ast, code_file, 0);
    pipeline::commented_code_generator::generate_code(ast, code_file, 0);
    code_file.close();
	return 0;
}