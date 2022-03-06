#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"

using builder::dyn_var;
using conv::TensorT;
using conv::ConvOptions;
using conv::PaddingT;

TensorT pad_input(TensorT input, TensorT weight, ConvOptions opt);
TensorT conv2d(TensorT input, TensorT weight, ConvOptions options);

void conv2d_nxn(dyn_var<int*> input, dyn_var<int*> weight, dyn_var<int*> output, dyn_var<int> input_size, dyn_var<int> weight_size, dyn_var<int> output_size);
