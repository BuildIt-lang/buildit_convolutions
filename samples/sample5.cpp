// tests for nd convolutions
#include <fstream>
#include "blocks/c_code_generator.h"
#include "builder/dyn_var.h"
#include "blocks/rce.h"
#include "conv_functions/conv2d.h"
#include "pipeline/conv.h"
#include "pipeline/conv_code_generator.h"

using builder::dyn_var;
using builder::static_var;
using conv::LoopSchedule;

int main() {
    std::ofstream code_file;
    code_file.open("./generated_code/test_convnd_code.h");
    code_file << "#include <assert.h>\n" << std::endl;
    code_file << "#include <omp.h>\n" << std::endl;
    
    int num_tests = 8;
    std::string func_name[] = {
        "conv3d_default_img5x5x5_ker3x3x3",
        "conv3d_default_img6x4x10_ker2x3x2",
        "conv3d_str2x3x3_img8x12x10_ker2x2x2",
        "conv3d_dil3x2x1_img12x10x8_ker2x1x2",
        "conv3d_pad2x4x3_img10x10x8_ker4x2x3",
        "conv3d_padsame_img10x9x9_ker3x3x3",
        "conv3d_str2x2x2_dil_2x3x2_pad_1x2x3_img_15x15x10_ker5x3x5",
        "conv3d_n3_ic4_oc5_str1x2x1_dil2x1x2_pad2x2x2_img10x6x8_ker2x2x2",
    };
    int stride[][3] = {{1, 1, 1}, {1, 1, 1}, {2, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {1, 2, 1}};
    int dilation[][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {3, 2, 1}, {1, 1, 1}, {1, 1, 1}, {2, 3, 2}, {2, 1, 2},};
    int padding[][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {2, 4, 3}, {0, 0, 0}, {1, 2, 3}, {2, 2, 2}};
    int img[][3] = {{5, 5, 5}, {6, 4, 10}, {8, 12, 10}, {12, 10, 8}, {10, 10, 8}, {10, 9, 9}, {15, 15, 10}, {10, 6, 8}};
    int ker[][3] = {{3, 3, 3}, {2, 3, 2}, {2, 2, 2}, {2, 1, 2}, {4, 2, 3}, {3, 3, 3}, {5, 3, 5}, {2, 2, 2}};
    int padding_same[] = {0, 0, 0, 0, 0, 1, 0, 0};
    int batch_size[] = {1, 1, 1, 1, 1, 1, 1, 3};
    int in_channels[] = {1, 1, 1, 1, 1, 1, 1, 4};
    int out_channels[] = {1, 1, 1, 1, 1, 1, 1, 5};
    for (int i = 0; i < num_tests; i ++) {
        LoopSchedule n = LoopSchedule(LoopSchedule::loop_type::N, batch_size[i]);
        LoopSchedule in_ch = LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]);
        LoopSchedule out_ch = LoopSchedule(LoopSchedule::loop_type::OC, out_channels[i]);
        LoopSchedule ix = LoopSchedule(LoopSchedule::loop_type::IMG, img[i][0]);
        LoopSchedule iy = LoopSchedule(LoopSchedule::loop_type::IMG, img[i][1]);
        LoopSchedule iz = LoopSchedule(LoopSchedule::loop_type::IMG, img[i][2]);
        LoopSchedule kx = LoopSchedule(LoopSchedule::loop_type::KERNEL, ker[i][0]);
        LoopSchedule ky = LoopSchedule(LoopSchedule::loop_type::KERNEL, ker[i][1]);
        LoopSchedule kz = LoopSchedule(LoopSchedule::loop_type::KERNEL, ker[i][2]);
        kx.dim = 0;
        ky.dim = 1;
        kz.dim = 2;
        ix.dim = 0;
        iy.dim = 1;
        iz.dim = 2;

        int ox = (img[i][0] - dilation[i][0] * (ker[i][0] - 1) - 1) / stride[i][0] + 1;
        int oy = (img[i][1] - dilation[i][1] * (ker[i][1] - 1) - 1) / stride[i][1] + 1;
        int oz = (img[i][2] - dilation[i][2] * (ker[i][2] - 1) - 1) / stride[i][2] + 1;
        ix.bound = ox;
        iy.bound = oy;
        iz.bound = oz;

        int ker_dims[] = {ker[i][0], ker[i][1], ker[i][2]};
        int img_dims[] = {img[i][0], img[i][1], img[i][2]};
        int out_dims[3];
        int pad_dims[3];
        int padded_img_dims[3];

        // int dims[] = {2, 5};
        // LoopSchedule subloops[2] = {LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]), LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i])};
        // in_ch.tile(dims, 2, subloops, 10);
        LoopSchedule all_loops[9] = {n, out_ch, in_ch, ix, iy, iz, kx, ky, kz};
        Schedule s = Schedule(all_loops, 9, 3);
        auto ast = builder::builder_context().extract_function_ast(static_conv2d_with_scheduling<float>, func_name[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 3, out_dims, pad_dims, padded_img_dims);
        block::eliminate_redundant_vars(ast);
        pipeline::conv_code_generator::generate_code(ast, code_file, 0);
        code_file << "\n" << std::endl;

    }
    code_file.close();
	return 0;
}