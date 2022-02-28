#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"

using builder::dyn_var;
using conv::TensorT;

TensorT conv2d(TensorT input, TensorT weight, dyn_var<int> stride, dyn_var<int> padding, dyn_var<int> dilation, dyn_var<int> groups);

void conv2d_nxn(dyn_var<int*> input, dyn_var<int*> weight, dyn_var<int*> output, dyn_var<int> input_size, dyn_var<int> weight_size, dyn_var<int> output_size);
