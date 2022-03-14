#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"
#include "conv_functions/runtime.h"
#include "blocks/c_code_generator.h"
#include "conv_functions/conv2d.h"

using builder::dyn_var;
using conv::PaddingT;
using conv::ConvOptions;
using conv::ImageT;
using conv::KernelT; 

/**
 * Returns a padded input image. If padding = "same", it calculates the amount
 * of padding on each side, and then pads the input.
 * */
ImageT pad_input(ImageT input, KernelT weight, ConvOptions opt) {
    ImageT new_input;
    new_input.batch_size = input.batch_size;
    new_input.in_channels = input.in_channels;
    if (opt.padding.is_same) {
        // the output should be the same shape as the input
        // torch does not support padding "same" with stride other than 1
        conv::runtime::conv_assert(opt.stride[0] == 1 && opt.stride[1] == 1);
        new_input.height = input.height * opt.stride[0] - opt.stride[0] + opt.dilation[0] * (weight.height - 1) + 1;
        new_input.width = input.width * opt.stride[1] - opt.stride[1] + opt.dilation[1] * (weight.width - 1) + 1;
        opt.padding.values[0] = (new_input.height - input.height) / 2;
        opt.padding.values[1] = (new_input.width - input.width) / 2;
    } else {
        new_input.height = input.height + 2 * opt.padding.values[0];
        new_input.width = input.width + 2 * opt.padding.values[1];
    }
    dyn_var<int> pad_h = opt.padding.values[0];
    dyn_var<int> pad_w = opt.padding.values[1];
    new_input.data = conv::runtime::conv_malloc((int)sizeof(int)*new_input.width*new_input.height*new_input.batch_size*new_input.in_channels);
    dyn_var<int> new_idx;
    dyn_var<int> old_idx;
    builder::annotate("Comment: creating a padded image");
    for (dyn_var<int> bid = 0; bid < input.batch_size; bid = bid + 1) {
        for (dyn_var<int> in_ch = 0; in_ch < input.in_channels; in_ch = in_ch + 1) {
            for (dyn_var<int> i = 0; i < new_input.height; i = i + 1) {
                for (dyn_var<int> j = 0; j < new_input.width; j = j + 1) {
                    new_idx = bid * new_input.in_channels * new_input.width * new_input.height + in_ch * new_input.width * new_input.height + i * new_input.width + j;
                    old_idx = bid * input.in_channels * input.width * input.height + in_ch * input.width * input.height + (i - pad_h) * input.width + (j - pad_w);
                    if (i < pad_h || j < pad_w || i >= input.height + pad_h || j >= input.width + pad_w) {
                        new_input.data[new_idx] = 0; // zero padding
                    } else {
                        new_input.data[new_idx] = input.data[old_idx];
                    }
                }
            }
        }
    }
    return new_input;
    
}

ImageT conv2d(ImageT inp, KernelT weight, ConvOptions opt) {

    conv::runtime::conv_assert(inp.in_channels == weight.in_channels);
    conv::runtime::conv_assert(weight.height <= inp.height);
    conv::runtime::conv_assert(weight.width <= inp.width);

    // pad the input
    ImageT input;
    if (opt.padding.is_same || opt.padding.values[0] != 0 || opt.padding.values[1] != 0) {
        input = pad_input(inp, weight, opt);
    } else {
        input = inp;
    }

    ImageT output;
    output.height = (input.height - opt.dilation[0] * (weight.height - 1) - 1) / opt.stride[0] + 1;
    output.width = (input.width - opt.dilation[1] * (weight.width - 1) - 1) / opt.stride[1] + 1;
    output.in_channels = weight.out_channels;
    output.batch_size = input.batch_size;
    dyn_var<int> size = output.width * output.height * output.batch_size * output.in_channels;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(int));
    dyn_var<int> out_idx;
    dyn_var<int> in_idx;
    dyn_var<int> weight_idx;
    builder::annotate("Comment: looping over batches");
    for (dyn_var<int> bid = 0; bid < output.batch_size; bid = bid + 1) {
        builder::annotate("Comment: looping over out channels");
        for (dyn_var<int> out_ch = 0; out_ch < weight.out_channels; out_ch = out_ch + 1) {
            builder::annotate("Comment: looping over in channels");
            for (dyn_var<int> in_ch = 0; in_ch < input.in_channels; in_ch = in_ch + 1) {
                builder::annotate("Comment: looping over the output");
                for (dyn_var<int> h = 0; h < output.height; h = h + 1) {
                    for (dyn_var<int> w = 0; w < output.width; w = w + 1) {
                        out_idx =  bid * output.in_channels * output.height * output.width + out_ch * output.width * output.height + h * output.width + w;
                        // output.data[out_idx] = 0;
                        builder::annotate("Comment: looping over the kernel");
                        for (dyn_var<int> i = 0; i < weight.height; i = i + 1){
                            for (dyn_var<int> j = 0; j < weight.width; j = j + 1) {
                                in_idx = bid * input.in_channels * input.width * input.height + in_ch * input.width * input.height + (h * opt.stride[0] + i * opt.dilation[0]) * input.width 
                                            + (w * opt.stride[1] + j * opt.dilation[1]);
                                weight_idx = out_ch * weight.in_channels * weight.width * weight.height + in_ch * weight.width * weight.height + i * weight.width + j;
                                output.data[out_idx] = output.data[out_idx] + input.data[in_idx] * weight.data[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
}
