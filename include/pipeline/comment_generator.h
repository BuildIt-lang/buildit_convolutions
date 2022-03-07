#include "blocks/c_code_generator.h"

namespace pipeline {

class commented_code_generator : public block::c_code_generator {

public:
    using c_code_generator::visit;
	commented_code_generator(std::ostream &_oss) : c_code_generator(_oss) {}
    virtual void visit(block::for_stmt::Ptr);
    static void generate_code(block::block::Ptr ast, std::ostream &oss, int indent = 0) {
		commented_code_generator generator(oss);
		generator.curr_indent = indent;
		ast->accept(&generator);
		oss << std::endl;
	}
};

}