#ifndef CONV_RUNTIME_H
#define CONV_RUNTIME_H

#include "builder/static_var.h"
#include "builder/dyn_var.h"

using builder::dyn_var;
using builder::static_var;

namespace conv {
namespace runtime {
    extern dyn_var<int*(int)> conv_malloc;
    extern dyn_var<void(int*)> conv_free;

    extern dyn_var<void (void)> start_timer;
    extern dyn_var<float (void)> stop_timer;
    extern dyn_var<void (float)> print_time;

    extern dyn_var<void (int*, int)> print_matrix;
}
}

#endif