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

ImageT static_conv2d(dyn_var<int*> inp_data, dyn_var<int*> weight_data, static_var<int> iw, static_var<int> ih, static_var<int> ww, static_var<int> wh, 
                    static_var<int> batch_size, static_var<int> in_channels, static_var<int> out_channels, static_var<int*> stride, static_var<int*> dilation, 
                    static_var<int*> padding, static_var<int> padding_same);
