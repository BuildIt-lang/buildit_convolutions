#include "conv_functions/runtime.h"

namespace conv {
namespace runtime {
    dyn_var<int*(int)> conv_malloc("conv_runtime::conv_malloc");
    dyn_var<void(int*)> conv_free("conv_runtime::conv_free");
}
}