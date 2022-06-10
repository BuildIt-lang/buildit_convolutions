// generates code for testing
#include <fstream>
#include "blocks/c_code_generator.h"
#include "builder/dyn_var.h"
#include "blocks/rce.h"
#include "conv_functions/conv2d.h"
#include "pipeline/conv.h"
#include "pipeline/conv_code_generator.h"

using builder::dyn_var;
using builder::static_var;

int main() {
    std::ofstream code_file;
    code_file.open("./generated_code/specialized_test_code.h");
    // code_file << "#include \"runtime_functions.h\"" << std::endl;
    // code_file << "#include \"runtime_types.h\"\n" << std::endl;
    code_file << "#include <assert.h>\n" << std::endl;
    code_file << "#include <omp.h>\n" << std::endl;
    
    int num_tests = 19;
    std::string func_name[] = {
        "conv2d_default_im5x5_w3x3", 
        "conv2d_stride2x1_im8x10_w3x2",
        "conv2d_dil3x2_im20x15_w3x2",
        "conv2d_stride2x3_dil3x2_im20x15_w3x2",
        "conv2d_pad1x2_im5x5_w3x2",
        "conv2d_padsame_im5x5_w3x3",
        "conv2d_dil3x2_stride2x3_pad3x4_im15x20_w3x2",
        "conv2d_dil3x2_padsame_im15x20_w3x3",
        "conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5",
        "conv2d_dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5",
        "conv2d_im100x100_w10x10_batch10_inch10_outch10",
        "conv2d_stride4x4_im100x100_w10x10_batch10_inch5_outch10",
        "conv2d_im10x10_w5x5_batch10_inch5_outch1",
        "conv2d_pad10x10_dil4x4_im10x10_w3x3_batch1_inch5_outch3",
        "conv2d_default_im5x2_w2x2",
        "conv2d_default_im10x10_w6x3",
        "conv2d_default_im1x6_w1x2",
        "conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_inch5",
        "conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_outch6",
        };
    int stride[][2] = {{1, 1}, {2, 1}, {1, 1}, {2, 3}, {1, 1}, {1, 1}, {2, 3}, {1, 1}, {2, 4}, {2, 4}, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {2, 4}, {2, 4}};
    int dilation[][2] = {{1, 1}, {1, 1}, {3, 2}, {3, 2}, {1, 1}, {1, 1}, {3, 2}, {3, 2}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, {4, 4}, {1, 1}, {1, 1}, {1, 1}, {2, 2}, {2, 2}};
    int padding[][2] ={{0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 2}, {0, 0}, {3, 4}, {0, 0}, {5, 4}, {5, 4}, {0, 0}, {0, 0}, {0, 0}, {10, 10}, {0, 0}, {0, 0}, {0, 0}, {5, 4}, {5, 4}};
    int padding_same[] = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int iw[] = {5, 10, 15, 15, 5, 5, 15, 15, 20, 20, 100, 100, 10, 10, 5, 10, 1, 20, 20};
    int ih[] = {5, 8, 20, 20, 5, 5, 20, 20, 20, 20, 100, 100, 10, 10, 2, 10, 6, 20, 20};
    int ww[] = {3, 2, 2, 2, 2, 3, 2, 3, 3, 5, 10, 10, 5, 3, 2, 6, 1, 3, 3};
    int wh[] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 10, 10, 5, 3, 2, 3, 2, 3, 3};
    int batch_size[] = {1, 1, 1, 1, 1, 1, 1, 1, 5, 4, 10, 10, 10, 1, 1, 1, 1, 1, 1};
    int in_channels[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 10, 5, 5, 5, 1, 1, 1, 5, 1};
    int out_channels[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 10, 10, 1, 3, 1, 1, 1, 1, 6};
    for (int i = 0; i < num_tests; i ++) {
        auto ast = builder::builder_context().extract_function_ast(static_conv2d_with_tiled_loops, func_name[i], iw[i], ih[i], ww[i], wh[i], batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i]);
        block::eliminate_redundant_vars(ast);
        pipeline::conv_code_generator::generate_code(ast, code_file, 0);
        code_file << "\n" << std::endl;
    }
    code_file.close();
	return 0;
}