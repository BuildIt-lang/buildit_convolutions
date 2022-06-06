#include "blocks/c_code_generator.h"
#include "pipeline/conv_code_generator.h"
#include <string.h>

using namespace block;

namespace pipeline {

void conv_code_generator::visit(for_stmt::Ptr a) {
    if (a->annotation == "") {
        c_code_generator::visit(a);
        return;
    }
    // get the left most annotation
    int delim_idx = (a->annotation).find("|");
    std::string curr_annotation;
    if (delim_idx != -1) {
        curr_annotation = (a->annotation).substr(0, delim_idx);
        a->annotation = (a->annotation).substr(delim_idx + 1);
    } else {
        curr_annotation = a->annotation;
        a->annotation = "";
    }
    int pragma_idx = curr_annotation.find("#pragma");
    int comment_idx = curr_annotation.find("Comment: ");
    // insert code based on the current annotation
    if (comment_idx != -1) {
        oss << "// " << (curr_annotation).substr(comment_idx + 9) << std::endl;
        printer::indent(oss, curr_indent);
        conv_code_generator::visit(a);
    } else if (pragma_idx != -1) {
        oss << curr_annotation.substr(pragma_idx) << std::endl;
        printer::indent(oss, curr_indent);
        conv_code_generator::visit(a);
    } else  {
        c_code_generator::visit(a);
    }
}

}
