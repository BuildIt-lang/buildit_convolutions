#ifndef PIPELINE_CONV_H
#define PIPELINE_CONV_H

#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

namespace pipeline {
    extern void generate_conv_code(block::block::Ptr ast, std::ofstream &out_file, std::string func_name);
}

#endif