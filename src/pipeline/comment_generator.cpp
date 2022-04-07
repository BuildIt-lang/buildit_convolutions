#include "blocks/c_code_generator.h"
#include "pipeline/comment_generator.h"
#include <string.h>

using namespace block;

namespace pipeline {

void commented_code_generator::visit(for_stmt::Ptr a) {
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
    // insert code based on the current annotation
    if (curr_annotation.find("Comment:") != std::string::npos) {
        oss << "// " << (curr_annotation).substr(9) << std::endl;
        printer::indent(oss, curr_indent);
        commented_code_generator::visit(a);
    } else if (curr_annotation.find("parallel block") != std::string::npos) {
        oss << "#pragma omp parallel" << std::endl;
        printer::indent(oss, curr_indent);
        oss << "{" << std::endl;
        printer::indent(oss, curr_indent);
        commented_code_generator::visit(a);
        oss << std::endl;
        oss << "}" << std::endl;
    } else if (curr_annotation.find("parallel for") != std::string::npos) {
        oss << "#pragma " << curr_annotation << std::endl;
        printer::indent(oss, curr_indent);
        commented_code_generator::visit(a);
    } else  {
        c_code_generator::visit(a);
    }
}

}
