#ifndef CONV_RUNTIME_H
#define CONV_RUNTIME_H

#include "builder/static_var.h"
#include "builder/dyn_var.h"

using builder::dyn_var;
using builder::static_var;

namespace conv {
namespace runtime {
    extern dyn_var<void*(int)> malloc;
    extern dyn_var<void(int*)> free;
}
}

#endif