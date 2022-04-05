#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"
#include "conv_functions/runtime.h"
#include "blocks/c_code_generator.h"
#include "conv_functions/conv2d.h"

using builder::dyn_var;
using builder::static_var;
using conv::PaddingT;
using conv::ConvOptions;
using conv::ImageT;
using conv::KernelT; 


ImageT dyn_conv2d(ImageT input, KernelT weight, ConvOptions opt) {

    conv::runtime::conv_assert(input.in_channels == weight.in_channels);
    conv::runtime::conv_assert(weight.height <= input.height);
    conv::runtime::conv_assert(weight.width <= input.width);

    
    dyn_var<int> pad_h = opt.padding.values[0];
    dyn_var<int> pad_w = opt.padding.values[1];
    dyn_var<int> ih;
    dyn_var<int> iw;
    if (opt.padding.is_same) {
        // torch does not support padding "same" with stride other than 1
        conv::runtime::conv_assert(opt.stride[0] == 1 && opt.stride[1] == 1);
        // calculate padding such that the output is the same shape as the input
        ih = input.height * opt.stride[0] - opt.stride[0] + opt.dilation[0] * (weight.height - 1) + 1;
        iw = input.width * opt.stride[1] - opt.stride[1] + opt.dilation[1] * (weight.width - 1) + 1;
        pad_h = (ih - input.height) / 2;
        pad_w = (iw - input.width) / 2;
    } else {
        ih = input.height + 2 * pad_h;
        iw = input.width + 2 * pad_w;
    }

    ImageT output;
    output.height = (ih - opt.dilation[0] * (weight.height - 1) - 1) / opt.stride[0] + 1;
    output.width = (iw - opt.dilation[1] * (weight.width - 1) - 1) / opt.stride[1] + 1;
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
                        
                                dyn_var<int> im_i = h * opt.stride[0] + i * opt.dilation[0];
                                dyn_var<int> im_j = w * opt.stride[1] + j * opt.dilation[1];
                                dyn_var<int> img_val;
                                if (im_i < pad_h || im_j < pad_w || im_i >= input.height + pad_h || im_j >= input.width + pad_w) {
                                    img_val = 0;
                                } else {
                                    img_val = input.data[bid * input.in_channels * input.width * input.height + in_ch * input.width * input.height + (im_i - pad_h) * input.width + (im_j - pad_w)];
                                }
                                weight_idx = out_ch * weight.in_channels * weight.width * weight.height + in_ch * weight.width * weight.height + i * weight.width + j;
                                output.data[out_idx] = output.data[out_idx] + img_val * weight.data[weight_idx];
                            }
                        }
                        // output.print();
                    }
                }
            }
        }
    }
    return output;
}

ImageT static_conv2d(dyn_var<int*> inp_data, dyn_var<int*> weight_data, int orig_iw, int orig_ih, int ww, int wh, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same) {

    conv::runtime::conv_assert(wh <= orig_ih);
    conv::runtime::conv_assert(ww <= orig_iw);

    static_var<int> pad_h = padding[0];
    static_var<int> pad_w = padding[1];
    static_var<int> ih;
    static_var<int> iw;
    if (padding_same == 1) {
        // torch does not support padding "same" with stride other than 1
        conv::runtime::conv_assert(stride[0] == 1 && stride[1] == 1);
        // calculate padding such that the output is the same shape as the input
        ih = orig_ih * stride[0] - stride[0] + dilation[0] * (wh - 1) + 1;
        iw = orig_iw * stride[1] - stride[1] + dilation[1] * (ww - 1) + 1;
        pad_h = (ih - orig_ih) / 2;
        pad_w = (iw - orig_iw) / 2;
    } else {
        ih = orig_ih + 2 * pad_h;
        iw = orig_iw + 2 * pad_w;
    }

    static_var<int> oh = (ih - dilation[0] * (wh - 1) - 1) / stride[0] + 1;
    static_var<int> ow = (iw - dilation[1] * (ww - 1) - 1) / stride[1] + 1;

    ImageT output;
    output.height = oh;
    output.width = ow;
    output.in_channels = out_channels;
    output.batch_size = batch_size;
    static_var<int> size = ow * oh * batch_size * out_channels;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(int));
    dyn_var<int> out_idx;
    dyn_var<int> weight_idx;
    builder::annotate("Comment: looping over batches");
    for (dyn_var<int> bid = 0; bid < batch_size; bid = bid + 1) {
        builder::annotate("Comment: looping over out channels");
        for (dyn_var<int> out_ch = 0; out_ch < out_channels; out_ch = out_ch + 1) {
            builder::annotate("Comment: looping over in channels");
            for (dyn_var<int> in_ch = 0; in_ch < in_channels; in_ch = in_ch + 1) {
                builder::annotate("Comment: looping over the output");
                for (dyn_var<int> h = 0; h < output.height; h = h + 1) {
                    for (dyn_var<int> w = 0; w < output.width; w = w + 1) {
                        out_idx =  bid * output.in_channels * output.height * output.width + out_ch * output.width * output.height + h * output.width + w;
                        builder::annotate("Comment: looping over the kernel");
                        for (dyn_var<int> i = 0; i < wh; i = i + 1) {
                            dyn_var<int> im_i = h * stride[0] + i * dilation[0];
                            if (im_i < pad_h) continue;
                            else if (im_i < orig_ih + pad_h) {
                                for (dyn_var<int> j = 0; j < ww; j = j + 1) {
                                    dyn_var<int> im_j = w * stride[1] + j * dilation[1];
                                    if (im_j < pad_w) continue;
                                    else if (im_j < orig_iw + pad_w) {
                                        dyn_var<int> img_val = inp_data[bid * in_channels * orig_iw * orig_ih + in_ch * orig_iw * orig_ih + (im_i - pad_h) * orig_iw + (im_j - pad_w)];
                                        weight_idx = out_ch * in_channels * ww * wh + in_ch * ww * wh + i * ww + j;
                                        output.data[out_idx] = output.data[out_idx] + img_val * weight_data[weight_idx];
                                    }
                                    else break;
                                }
                            }
                            else break;
                        }
                        
                    }
                }
            }
        }
    }
    return output;
}

dyn_var<int> min(dyn_var<int> a, dyn_var<int> b) {
    if (a < b) return a;
    return b;
}

/*
ImageT static_conv2d(dyn_var<int*> inp_data, dyn_var<int*> weight_data, int orig_iw, int orig_ih, int ww, int wh, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same) {

    conv::runtime::conv_assert(wh <= orig_ih);
    conv::runtime::conv_assert(ww <= orig_iw);

    static_var<int> pad_h = padding[0];
    static_var<int> pad_w = padding[1];
    static_var<int> ih;
    static_var<int> iw;
    if (padding_same == 1) {
        // torch does not support padding "same" with stride other than 1
        conv::runtime::conv_assert(stride[0] == 1 && stride[1] == 1);
        // calculate padding such that the output is the same shape as the input
        ih = orig_ih * stride[0] - stride[0] + dilation[0] * (wh - 1) + 1;
        iw = orig_iw * stride[1] - stride[1] + dilation[1] * (ww - 1) + 1;
        pad_h = (ih - orig_ih) / 2;
        pad_w = (iw - orig_iw) / 2;
    } else {
        ih = orig_ih + 2 * pad_h;
        iw = orig_iw + 2 * pad_w;
    }

    static_var<int> dil_wh = dilation[0] * (wh - 1) + wh;
    static_var<int> dil_ww = dilation[1] * (ww - 1) + ww;

    static_var<int> oh = (ih - dilation[0] * (wh - 1) - 1) / stride[0] + 1;
    static_var<int> ow = (iw - dilation[1] * (ww - 1) - 1) / stride[1] + 1;

    ImageT output;
    output.height = oh;
    output.width = ow;
    output.in_channels = out_channels;
    output.batch_size = batch_size;
    static_var<int> size = ow * oh * batch_size * out_channels;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(int));
    dyn_var<int> out_idx;
    dyn_var<int> weight_idx;
    builder::annotate("Comment: looping over batches");
    for (dyn_var<int> bid = 0; bid < batch_size; bid = bid + 1) {
        builder::annotate("Comment: looping over out channels");
        for (dyn_var<int> out_ch = 0; out_ch < out_channels; out_ch = out_ch + 1) {
            builder::annotate("Comment: looping over in channels");
            for (dyn_var<int> in_ch = 0; in_ch < in_channels; in_ch = in_ch + 1) {
                builder::annotate("Comment: looping over the output");
                for (dyn_var<int> h = 0; h < output.height; h = h + 1) {
                    for (dyn_var<int> w = 0; w < output.width; w = w + 1) {
                        out_idx =  bid * output.in_channels * output.height * output.width + out_ch * output.width * output.height + h * output.width + w;
                        dyn_var<int> img_val;
                        dyn_var<int> ki;
                        dyn_var<int> kj;
                        dyn_var<int> bound_i = pad_h - h * stride[0];
                        dyn_var<int> bound_j = pad_w - w * stride[1];

                        builder::annotate("Comment: looping over the kernel");
                        for (ki = 0; ki < min(bound_i, wh * dilation[0]); ki = ki + dilation[0]) {
                            // if (ki >= wh * dilation[0]) break;
                            output.data[out_idx] = 0;
                        }
                        for (; ki < min(orig_ih + bound_i, wh * dilation[0]); ki = ki + dilation[0]) {
                            // if (ki >= wh * dilation[0]) break;
                            for (kj = 0; kj < min(bound_j, ww * dilation[1]); kj = kj + dilation[1]) {
                                // if (kj >= ww * dilation[1]) break;
                                output.data[out_idx] = 0;
                            }
                            for (; kj < min(orig_iw + bound_j, ww * dilation[1]); kj = kj + dilation[1]) {
                                // if (kj >= ww * dilation[1]) break;
                                img_val = inp_data[bid * in_channels * orig_iw * orig_ih + in_ch * orig_iw * orig_ih + (ki - bound_i) * orig_iw + (kj - bound_j)];
                                weight_idx = out_ch * in_channels * ww * wh + in_ch * ww * wh + (ki / dilation[0]) * ww + (kj / dilation[1]);
                                output.data[out_idx] = output.data[out_idx] + img_val * weight_data[weight_idx];
                            }
                            for (; kj < min(iw - w * stride[1], ww * dilation[1]); kj = kj +dilation[1]) {
                                // if (kj >= ww * dilation[1]) break;
                                output.data[out_idx] = 0;
                            }
                        }
                        for (; ki < min(ih - h * stride[0], wh * dilation[0]); ki = ki + dilation[0]) {
                            // if (ki >= wh * dilation[0]) break;
                            // img_val = 0;
                            output.data[out_idx] = 0;
                        }
                        
                    }
                }
            }
        }
    }
    return output;
}
*/

