#include <fstream>
#include "blocks/c_code_generator.h"
#include "builder/dyn_var.h"
#include "blocks/rce.h"
#include "conv_functions/conv2d.h"
#include "pipeline/conv.h"
#include "pipeline/comment_generator.h"

using builder::dyn_var;
using builder::static_var;

int main() {
    std::ofstream code_file;
    code_file.open("./generated_code/specialized_timing_code.h");
    code_file << "#include <assert.h>" << std::endl;
    code_file << "#include <omp.h>" << std::endl;

    int n_runs = 9;
    int iw[] = {100, 100, 200, 200, 300, 300, 200, 200, 128};
    int ih[] = {100, 100, 200, 200, 300, 300, 200, 200, 128};
    int kw[] = {10, 10, 10, 10, 10, 10, 10, 10, 5};
    int kh[] = {10, 10, 10, 10, 10, 10, 10, 10, 5};
    int batch_size[] = {10, 20, 10, 20, 10, 20, 10, 10, 32};
    int in_channels[] = {10, 10, 10, 10, 10, 10, 10, 10, 64};
    int out_channels[] = {10, 10, 10, 10, 10, 10, 10, 10, 16};
    int stride[][2] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {5, 5}, {1, 1}};
    int padding[][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {10, 10}, {10, 10}, {0, 0}};
    int dilation[][2] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    int padding_same[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::string func_names[] = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"};
    for (int i = 0; i < n_runs; i ++) {
        LoopSchedule n = LoopSchedule(LoopSchedule::loop_type::N, batch_size[i]);
        LoopSchedule in_ch = LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]);
        LoopSchedule out_ch = LoopSchedule(LoopSchedule::loop_type::OC, out_channels[i]);
        LoopSchedule iy = LoopSchedule(LoopSchedule::loop_type::IMG, ih[i]);
        LoopSchedule ix = LoopSchedule(LoopSchedule::loop_type::IMG, iw[i]);
        LoopSchedule ky = LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]);
        LoopSchedule kx = LoopSchedule(LoopSchedule::loop_type::KERNEL, kw[i]);
        ky.dim = 0;
        kx.dim = 1;
        ix.dim = 1;
        iy.dim = 0;

        int oh = (ih[i] - dilation[i][0] * (kh[i] - 1) - 1) / stride[i][0] + 1;
        int ow = (iw[i] - dilation[i][1] * (kw[i] - 1) - 1) / stride[i][1] + 1;
        ix.bound = ow;
        iy.bound = oh;

        if (i == 0) {
            n.parallelize(3);
            int ky_dims[] = {5, 2};
            LoopSchedule ky_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i])};
            ky.tile(ky_dims, 2, ky_loops, 10);
            int ic_dims[] = {2, 5};
            LoopSchedule ic_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]), LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i])};
            in_ch.tile(ic_dims, 2, ic_loops, 10);
            LoopSchedule all_loops[9] = {n, out_ch, iy, ic_loops[0], ky_loops[0], kx, ic_loops[0], ix, ky_loops[1]};
            Schedule s = Schedule(all_loops, 9);
            auto ast = builder::builder_context().extract_function_ast(static_conv2d_with_scheduling, func_names[i], iw[i], ih[i], kw[i], kh[i], batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s);
            block::eliminate_redundant_vars(ast);
            pipeline::commented_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else if (i == 1) {
            n.parallelize(3);
            int iy_dims[] = {13, 7};
            LoopSchedule iy_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IMG, ih[i]), LoopSchedule(LoopSchedule::loop_type::IMG, ih[i])};
            iy.tile(iy_dims, 2, iy_loops, 91);
            int ky_dims[] = {5, 2};
            LoopSchedule ky_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i])};
            ky.tile(ky_dims, 2, ky_loops, 10);
            int ic_dims[] = {5, 2};
            LoopSchedule ic_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]), LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i])};
            in_ch.tile(ic_dims, 2, ic_loops, 10);
            LoopSchedule all_loops[10] = {n, out_ch, ic_loops[0], ky_loops[0], iy_loops[0], iy_loops[1], ky_loops[1], ix, kx, ic_loops[1]};
            Schedule s = Schedule(all_loops, 10);
            auto ast = builder::builder_context().extract_function_ast(static_conv2d_with_scheduling, func_names[i], iw[i], ih[i], kw[i], kh[i], batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s);
            block::eliminate_redundant_vars(ast);
            pipeline::commented_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else {
            auto ast = builder::builder_context().extract_function_ast(static_conv2d_with_tiled_loops, func_names[i], iw[i], ih[i], kw[i], kh[i], batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i]);
            block::eliminate_redundant_vars(ast);
            pipeline::commented_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        }
    }
    code_file.close();
	return 0;
}
