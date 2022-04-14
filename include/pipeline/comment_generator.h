#include "blocks/c_code_generator.h"
#include "builder/builder_context.h"
#include "blocks/rce.h"
#include <string.h>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <dlfcn.h>

#include "gen/compiler_headers.h"

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

	template <typename FT, typename...ArgsT>
	static void* compile_function(FT f, ArgsT...args) {
		builder::builder_context context;	
		// Currently we will use an unconfigured context
		// Can take in extra parameters or a context object
		auto ast = context.extract_function_ast(f, "execute", args...);
		// Proactively run RCE
		block::eliminate_redundant_vars(ast);		

		char base_name_c[] = GEN_TEMPLATE_NAME;
		int fd = mkstemp(base_name_c);
		if (fd < 0) {
			assert(false && "Opening a temporary file failed\n");
		}
		
		close(fd);

		std::string base_name(base_name_c);	
		std::string source_name = base_name + ".c";
		std::string compiled_name = base_name + ".so";
		
		std::string compiler_name = COMPILER_PATH;
		std::string compile_command = compiler_name + " -shared -O3 " + source_name + " -o " + compiled_name;
		
			
		std::ofstream oss(source_name);	

		generate_code(ast, oss, 0);

		oss.close();

		int err = system(compile_command.c_str());
		if (err != 0) {	
			assert(false && "Compilation failed\n");
		}
		
		void* handle = dlopen(compiled_name.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (!handle) {
			assert(false && "Loading compiled module failed\n");
		}
		
		void* function = dlsym(handle, "execute");
		if (!function) {
			assert(false && "Loading compiled module failed\n");
		}
		return function;
	}
};

}