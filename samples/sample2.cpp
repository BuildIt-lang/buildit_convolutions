// Include the headers
#include "blocks/c_code_generator.h"
#include "builder/static_var.h"
#include "builder/dyn_var.h"
#include <iostream>

// Include the BuildIt types
using builder::dyn_var;
using builder::static_var;
static void bar(void) {
     // Insert code to stage here
}

int main(int argc, char* argv[]) {
	block::c_code_generator::generate_code(builder::builder_context().extract_function_ast(bar, "bar"), std::cout, 0);
	return 0;
}
