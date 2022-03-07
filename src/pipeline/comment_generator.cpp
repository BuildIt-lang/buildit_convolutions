#include "blocks/c_code_generator.h"
#include "pipeline/comment_generator.h"

using namespace block;

namespace pipeline {

void commented_code_generator::visit(for_stmt::Ptr a) {
    if ((a->annotation).find("Comment: ") == 0) {
        oss << "// " << (a->annotation).substr(9) << std::endl;
        printer::indent(oss, curr_indent);
    }
    c_code_generator::visit(a);
}

}