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
}
}

#endif