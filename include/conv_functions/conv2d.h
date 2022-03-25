#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"

using builder::dyn_var;
using builder::static_var;
using conv::ConvOptions;
using conv::PaddingT;
using conv::ImageT;
using conv::KernelT;

ImageT dyn_conv2d(ImageT input, KernelT weight, ConvOptions options);

ImageT static_conv2d(dyn_var<int*> inp_data, dyn_var<int*> weight_data, int iw, int ih, int ww, int wh, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same);
