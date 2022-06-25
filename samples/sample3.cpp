// the generated code is used for timing
// the schedules are taken from the result of tvm tuning
#include <fstream>
#include "blocks/c_code_generator.h"
#include "builder/dyn_var.h"
#include "blocks/rce.h"
#include "conv_functions/conv_nd.h"
#include "pipeline/conv.h"
#include "pipeline/conv_code_generator.h"

using builder::dyn_var;
using builder::static_var;

int main() {
    std::ofstream code_file;
    code_file.open("./generated_code/timing_code.h");
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
        int oh = (ih[i] - dilation[i][0] * (kh[i] - 1) - 1) / stride[i][0] + 1;
        int ow = (iw[i] - dilation[i][1] * (kw[i] - 1) - 1) / stride[i][1] + 1;
        LoopSchedule n = LoopSchedule(LoopSchedule::loop_type::N, batch_size[i]);
        LoopSchedule in_ch = LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]);
        LoopSchedule out_ch = LoopSchedule(LoopSchedule::loop_type::OC, out_channels[i]);
        LoopSchedule iy = LoopSchedule(LoopSchedule::loop_type::IMG, oh);
        LoopSchedule ix = LoopSchedule(LoopSchedule::loop_type::IMG, ow);
        LoopSchedule ky = LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]);
        LoopSchedule kx = LoopSchedule(LoopSchedule::loop_type::KERNEL, kw[i]);
        ky.dim = 0;
        kx.dim = 1;
        ix.dim = 1;
        iy.dim = 0;

        int ker_dims[] = {kh[i], kw[i]};
        int img_dims[] = {ih[i], iw[i]};

        if (i == 0) {
            n.parallelize(3);
            int ky_dims[] = {2, 5};
            LoopSchedule ky_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i])};
            ky.tile(ky_dims, 2, ky_loops, 10);
            int ic_dims[] = {5, 2};
            LoopSchedule ic_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]), LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i])};
            in_ch.tile(ic_dims, 2, ic_loops, 10);
            LoopSchedule all_loops[9] = {n, out_ch, iy, ic_loops[0], ky_loops[0], kx, ic_loops[1], ix, ky_loops[1]};
            Schedule s = Schedule(all_loops, 9, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
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
            Schedule s = Schedule(all_loops, 10, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else if (i == 2) {
            n.parallelize(3);
            int ky_dims[] = {2, 5};
            LoopSchedule ky_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i])};
            ky.tile(ky_dims, 2, ky_loops, 10);
            int kx_dims[] = {5, 2};
            LoopSchedule kx_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kw[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kw[i])};
            kx.tile(kx_dims, 2, kx_loops, 10);
            LoopSchedule all_loops[9] = {n, out_ch, ky_loops[0], kx_loops[0], iy, ix, ky_loops[1], in_ch, kx_loops[1]};
            Schedule s = Schedule(all_loops, 9, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else if (i == 3) {
            n.parallelize(3);
            int ic_dims[] = {5, 2};
            LoopSchedule ic_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]), LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i])};
            in_ch.tile(ic_dims, 2, ic_loops, 10);
            int ky_dims[] = {2, 5};
            LoopSchedule ky_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i])};
            ky.tile(ky_dims, 2, ky_loops, 10);
            int kx_dims[] = {5, 2};
            LoopSchedule kx_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kw[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kw[i])};
            kx.tile(kx_dims, 2, kx_loops, 10);
            LoopSchedule all_loops[10] = {n, out_ch, ky_loops[0], ic_loops[0], ix, kx_loops[0], kx_loops[1], ky_loops[1], ic_loops[1], iy};
            Schedule s = Schedule(all_loops, 10, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else if (i == 4) {
            n.parallelize(3);
            int iy_dims[] = {3, 97};
            LoopSchedule iy_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IMG, ih[i]), LoopSchedule(LoopSchedule::loop_type::IMG, ih[i])};
            iy.tile(iy_dims, 2, iy_loops, 291);
            int ky_dims[] = {2, 5};
            LoopSchedule ky_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i])};
            ky.tile(ky_dims, 2, ky_loops, 10);
            LoopSchedule all_loops[9] = {n, out_ch, kx, iy_loops[0], ky_loops[0], in_ch, iy_loops[1], ix, ky_loops[1]};
            Schedule s = Schedule(all_loops, 9, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else if (i == 5) {
            n.parallelize(3);
            int iy_dims[] = {97, 3};
            LoopSchedule iy_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IMG, ih[i]), LoopSchedule(LoopSchedule::loop_type::IMG, ih[i])};
            iy.tile(iy_dims, 2, iy_loops, 291);
            LoopSchedule all_loops[8] = {n, out_ch, iy_loops[0], ky, ix, in_ch, kx, iy_loops[1]};
            Schedule s = Schedule(all_loops, 8, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else if (i == 6) {
            n.parallelize(3);
            int ky_dims[] = {2, 5};
            LoopSchedule ky_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kh[i])};
            ky.tile(ky_dims, 2, ky_loops, 10);
            int kx_dims[] = {2, 5};
            LoopSchedule kx_loops[2] = {LoopSchedule(LoopSchedule::loop_type::KERNEL, kw[i]), LoopSchedule(LoopSchedule::loop_type::KERNEL, kw[i])};
            kx.tile(kx_dims, 2, kx_loops, 10);
            LoopSchedule all_loops[9] = {n, out_ch, kx_loops[0], ky_loops[0], iy, kx_loops[1], ky_loops[1], in_ch, ix};
            Schedule s = Schedule(all_loops, 9, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else if (i == 7) {
            n.parallelize(3);
            LoopSchedule all_loops[7] = {n, out_ch, in_ch, ky, iy, ix, kx};
            Schedule s = Schedule(all_loops, 7, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else if (i == 9) {
            n.parallelize(3);
            int ic_dims[] = {2, 8};
            LoopSchedule ic_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i]), LoopSchedule(LoopSchedule::loop_type::IC, in_channels[i])};
            in_ch.tile(ic_dims, 2, ic_loops, 16);
            int ix_dims[] = {2, 62};
            LoopSchedule ix_loops[2] = {LoopSchedule(LoopSchedule::loop_type::IMG, iw[i]), LoopSchedule(LoopSchedule::loop_type::IMG, iw[i])};
            kx.tile(ix_dims, 124, ix_loops, 10);
            LoopSchedule all_loops[9] = {n, out_ch, ic_loops[0], ic_loops[1], kx, iy, ky, ix_loops[0], ix_loops[1]};
            Schedule s = Schedule(all_loops, 9, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        } else {
            LoopSchedule all_loops[7] = {n, out_ch, in_ch, ky, kx, iy, ix};
            Schedule s = Schedule(all_loops, 7, 2);
            auto ast = builder::builder_context().extract_function_ast(conv_nd_main<float>, func_names[i], img_dims, ker_dims, batch_size[i], in_channels[i], out_channels[i], stride[i], dilation[i], padding[i], padding_same[i], s, 2);
            block::eliminate_redundant_vars(ast);
            pipeline::conv_code_generator::generate_code(ast, code_file, 0);
            code_file << "\n" << std::endl;
        }
    }
    code_file.close();
    return 0;
}
                                      