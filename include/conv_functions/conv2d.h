#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"

using builder::dyn_var;
using conv::TensorT;

TensorT conv2d(TensorT input, TensorT weight, int stride, int padding, int dilation, int groups);

void conv2d_nxn(dyn_var<int*> input, dyn_var<int*> weight, dyn_var<int*> output, dyn_var<int> input_size, dyn_var<int> weight_size, dyn_var<int> output_size);
