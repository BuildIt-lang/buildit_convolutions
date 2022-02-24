#include "conv_functions/runtime.h"

namespace conv {
namespace runtime {
    dyn_var<int*(int)> conv_malloc("conv_runtime::conv_malloc");
    dyn_var<void(int*)> conv_free("conv_runtime::conv_free");

    dyn_var<void (void)> start_timer("conv_runtime::start_timer");
    dyn_var<float (void)> stop_timer("conv_runtime::stop_timer");
    dyn_var<void (float)> print_time("conv_runtime::print_time");

    dyn_var<void (int*, int)> print_matrix("conv_runtime::print_matrix");
}
}