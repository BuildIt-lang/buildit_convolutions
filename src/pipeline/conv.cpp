#include "pipeline/conv.h"
#include <fstream>

namespace pipeline {
    
    void generate_conv_code(block::block::Ptr ast, std::ofstream &out_file) {
        
        out_file << "#include \"mem_allocation.h\"\n" << std::endl;
        block::c_code_generator::generate_code(ast, out_file);
    }

}