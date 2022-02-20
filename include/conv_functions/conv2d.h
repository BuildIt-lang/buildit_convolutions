#include "builder/static_var.h"
#include "builder/dyn_var.h"

using builder::dyn_var;
using builder::static_var;

void conv2d(dyn_var<int*> input, dyn_var<int*> weight, dyn_var<int*> output, dyn_var<int> input_size, dyn_var<int> weight_size, dyn_var<int> output_size);