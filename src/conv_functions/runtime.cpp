#include "conv_functions/runtime.h"

namespace conv {
namespace runtime {
    dyn_var<void*(int)> malloc("conv_runtime::malloc");
    dyn_var<void(int*)> free("conv_runtime::free");
}
}