#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"

using builder::dyn_var;
using builder::static_var;
using conv::ConvOptions;
using conv::PaddingT;
using conv::ImageT;
using conv::KernelT;

ImageT dyn_pad_input(ImageT input, KernelT weight, ConvOptions opt);
ImageT dyn_conv2d(ImageT input, KernelT weight, ConvOptions options);

ImageT static_pad_input(ImageT input, KernelT weight, int* stride, int* dilation, int* padding, int padding_same);
ImageT static_conv2d(ImageT inp, KernelT weight, static_var<int*> stride, static_var<int*> dilation, static_var<int*> padding, static_var<int> padding_same);
