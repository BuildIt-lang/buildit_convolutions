#include "pipeline/conv.h"
#include <fstream>

namespace pipeline {
    
    void generate_conv_code(block::block::Ptr ast, std::ofstream &out_file, std::string func_name) {
        block::eliminate_redundant_vars(ast);
        out_file << "#include \"runtime_functions.h\"\n" << std::endl;
        block::c_code_generator::generate_code(ast, out_file);
        out_file << "int main() {" << std::endl;
        out_file << "\t" << func_name << "();" << std::endl;
        out_file << "\treturn 0;" << std::endl;
        out_file << "}" << std::endl;
    }

}