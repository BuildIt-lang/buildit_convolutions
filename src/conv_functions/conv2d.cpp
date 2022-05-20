#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"
#include "conv_functions/runtime.h"
#include "conv_functions/schedule.h"
#include "blocks/c_code_generator.h"
#include "conv_functions/conv2d.h"

using builder::dyn_var;
using builder::static_var;
using conv::PaddingT;
using conv::ConvOptions;
using conv::ImageT;
using conv::KernelT; 
using conv::Schedule;
using conv::LoopSchedule;


ImageT<conv_t> dyn_conv2d(ImageT<conv_t> input, KernelT<conv_t> weight, ConvOptions opt) {

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

    ImageT<conv_t> output;
    output.height = (ih - opt.dilation[0] * (weight.height - 1) - 1) / opt.stride[0] + 1;
    output.width = (iw - opt.dilation[1] * (weight.width - 1) - 1) / opt.stride[1] + 1;
    output.in_channels = weight.out_channels;
    output.batch_size = input.batch_size;
    dyn_var<int> size = output.width * output.height * output.batch_size * output.in_channels;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(conv_t));
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

ImageT<conv_t> static_conv2d(dyn_var<conv_t*> inp_data, dyn_var<conv_t*> weight_data, int orig_iw, int orig_ih, int ww, int wh, 
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

    ImageT<conv_t> output;
    output.height = oh;
    output.width = ow;
    output.in_channels = out_channels;
    output.batch_size = batch_size;
    static_var<int> size = ow * oh * batch_size * out_channels;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(conv_t));
    builder::annotate("Comment: looping over batches | omp parallel for collapse(3)");
    for (dyn_var<int> bid = 0; bid < batch_size; bid = bid + 1) {
        builder::annotate("Comment: looping over out channels");
        for (dyn_var<int> out_ch = 0; out_ch < out_channels; out_ch = out_ch + 1) {
            builder::annotate("Comment: looping over in channels");
            for (dyn_var<int> in_ch = 0; in_ch < in_channels; in_ch = in_ch + 1) {
                dyn_var<int> out_idx;
                dyn_var<int> weight_idx;
                dyn_var<int> counter = 0;
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
                                        counter = counter + 1;
                                    }
                                    else break;
                                }
                            }
                            else break;
                        }
                        
                    }
                }
                output.mult_cnt = counter; // should be safe to write in parallel since it's the same number for all iters
            }
        } 
    }
    output.mult_cnt = output.mult_cnt * batch_size * in_channels * out_channels;
    return output;
}

/**
 * Splits the image row/column ranges into 6 regions based on the location of the kernel.
 * Region 0: the kernel is completely in the left/upper padded area
 * Region 1: the kernel intersects only the left/upper border between the padded area and the original image
 * Region 2: completely inside the original image
 * Region 3: intersects only the right/lower border
 * Region 4: intersects both the left/upper and the right/lower border
 * Region 5: completely inside the right/lower padded area
 */

void get_bounds(int* img_bounds, int* ker_bounds, int out_size, int ker_size, int pad, int stride, int dilation, int orig_size, int im_size) {
    static_var<int> curr = -1;
    static_var<int> prev_i_lo = 0;
    static_var<int> prev_i_hi = -1;
    // init all img bounds to empty ranges
    for (static_var<int> k = 0; k < 6; k = k + 1) {
        img_bounds[k * 2] = -1;
        img_bounds[k * 2 + 1] = -2;
        ker_bounds[k * 2] = -1;
        ker_bounds[k * 2 + 1] = -2;
    }
    for (static_var<int> h = 0; h < out_size; h = h + 1) {
        static_var<int> im_idx_lo = h * stride;
        static_var<int> im_idx_hi = h * stride + (ker_size - 1) * dilation;
        if (im_idx_hi < pad) { // left region 0
            if (curr != 0) {
                curr = 0;
                img_bounds[curr * 2] = 0;
            }
        } else if (im_idx_lo < pad && im_idx_hi < orig_size + pad) { // intersecting left border 1
            if (curr != 1) {
                if (curr != -1) img_bounds[curr * 2 + 1] = h;
                curr = 1;
                img_bounds[curr * 2] = h;
                ker_bounds[curr * 2] = -1; // kernel start idx depends on h
                ker_bounds[curr * 2 + 1] = ker_size;
            }
        } else if (im_idx_lo >= pad && im_idx_hi < orig_size + pad) { // fits completely in center 2
            if (curr != 2) {
                if (curr != -1) img_bounds[curr * 2 + 1] = h;
                curr = 2;
                img_bounds[curr * 2] = h;
                ker_bounds[curr * 2] = 0; // the entire dilated kernel fits
                ker_bounds[curr * 2 + 1] = ker_size;
            }
        } else if (im_idx_lo >= pad && im_idx_hi >= orig_size + pad) { // intersecting right border 3
            if (curr != 3) {
                if (curr != -1) img_bounds[curr * 2 + 1] = h;
                curr = 3;
                img_bounds[curr * 2] = h;
                ker_bounds[curr * 2] = 0;
                ker_bounds[curr * 2 + 1] = -1; // kernel end idx depends on h
            }
        } else if (im_idx_lo < pad && im_idx_hi >= orig_size + pad) { // intersecting both left and right border 4
             if (curr != 4) {
                if (curr != -1) img_bounds[curr * 2 + 1] = h;
                curr = 4;
                img_bounds[curr * 2] = h;
                ker_bounds[curr * 2] = -1; // kernel start idx depends on h
                ker_bounds[curr * 2 + 1] = -1; // kernel end idx depends on h
            }
        } else if (im_idx_lo >= orig_size + pad) { // right region 5
             if (curr != 5) {
                if (curr != -1) img_bounds[curr * 2 + 1] = h;
                curr = 5;
                img_bounds[curr * 2] = h;
            }
        } 
    }
    if (curr != -1) img_bounds[curr * 2 + 1] = out_size;
}

/**
 * Computes an output image value for a specific index.
 */
void update_output(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, dyn_var<int> out_idx,
            dyn_var<int> im_i, dyn_var<int> im_j, dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, dyn_var<int> i, dyn_var<int> j,
            int orig_iw, int ww, int pad_h, int pad_w) {
    dyn_var<int> img_val = input_data[inner_img_idx + (im_i - pad_h) * orig_iw + (im_j - pad_w)];
    dyn_var<int> weight_idx = inner_ker_idx + i * ww + j;
    output_data[out_idx] = output_data[out_idx] + img_val * weight_data[weight_idx];
}

/**
 * Loops over kernel columns.
 */
dyn_var<int> kernel_w_loop(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, dyn_var<int> w_stride, 
    dyn_var<int> out_idx, dyn_var<int> im_i, dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, int ww, 
    dyn_var<int> i, int orig_iw, int pad_h, int pad_w, int dil, bool h_condition) {
    dyn_var<int> counter = 0;
    builder::annotate("Comment: looping over kernel columns");
    for (dyn_var<int> j = 0; j < ww; j = j + 1) {
        dyn_var<int> im_j = w_stride + j * dil;
        if (h_condition) {
            if (im_j < pad_w) continue;
            else if (im_j < orig_iw + pad_w) {
                update_output(input_data, weight_data, output_data, out_idx,
                    im_i, im_j, inner_img_idx, inner_ker_idx, i, j, orig_iw, ww, pad_h, pad_w);
                counter = counter + 1;
                
            }
            else break;
        } else {
            update_output(input_data, weight_data, output_data, out_idx,
                im_i, im_j, inner_img_idx, inner_ker_idx, i, j, orig_iw, ww, pad_h, pad_w);
            counter = counter + 1;
        }
    }
    return counter;
}

/**
 * Loops over the kernel.
 * There are if conditions when the kernel intersects
 * the border between the padded area and the original image.
 */
dyn_var<int> kernel_loops(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data,
            dyn_var<int> h, dyn_var<int> w, dyn_var<int> w_stride, dyn_var<int> h_stride, dyn_var<int> out_idx, 
            dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, int ww, int wh,
            int* dilation, int pad_h, int pad_w,
            int orig_ih, int orig_iw, int in_channels, int out_channels, bool w_condition, bool h_condition) {
    dyn_var<int> counter = 0;
    builder::annotate("Comment: looping over kernel rows");
    for (dyn_var<int> i = 0; i < wh; i = i + 1) {
        dyn_var<int> im_i = h_stride + i * dilation[0];
        if (w_condition) {
            if (im_i < pad_h) {
                continue;
            } else if (im_i < orig_ih + pad_h) {
                counter = counter + kernel_w_loop(input_data, weight_data, output_data, w_stride, out_idx, im_i, inner_img_idx, inner_ker_idx, ww, 
                    i, orig_iw, pad_h, pad_w, dilation[1], h_condition);
            } else break;
        } else {
            counter = counter + kernel_w_loop(input_data, weight_data, output_data, w_stride, out_idx, im_i, inner_img_idx, inner_ker_idx, ww, 
                i, orig_iw, pad_h, pad_w, dilation[1], h_condition);
        }
    }
    return counter;
}

/**
 * Same as static_conv2d but splits the image loops based on
 * kernel location. Currently works only for padding value 0.
 */
ImageT<conv_t> static_conv2d_with_tiled_loops(dyn_var<conv_t*> inp_data, dyn_var<conv_t*> weight_data, int orig_iw, int orig_ih, int ww, int wh, 
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
    static_var<int> oh_times_ow = oh * ow;
    static_var<int> inch_oh_ow = out_channels * oh_times_ow;

    static_var<int> orig_h_w = orig_ih * orig_iw;
    static_var<int> orig_inch_h_w = in_channels * orig_h_w;
    static_var<int> ker_w_h = ww * wh;
    static_var<int> ker_inch_w_h = in_channels * ker_w_h;

    ImageT<conv_t> output;
    output.height = oh;
    output.width = ow;
    output.in_channels = out_channels;
    output.batch_size = batch_size;
    static_var<int> size = ow * oh * batch_size * out_channels;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(conv_t));
    builder::annotate("Comment: looping over batches | #pragma omp parallel for collapse(3)");
    for (dyn_var<int> bid = 0; bid < batch_size; bid = bid + 1) {
        builder::annotate("Comment: looping over out channels");
        for (dyn_var<int> out_ch = 0; out_ch < out_channels; out_ch = out_ch + 1) {
            builder::annotate("Comment: looping over in channels");
            for (dyn_var<int> in_ch = 0; in_ch < in_channels; in_ch = in_ch + 1) {
                dyn_var<int> out_idx;
                dyn_var<int> weight_idx;
                dyn_var<int> counter = 0;

                dyn_var<int> inner_img_idx = bid * orig_inch_h_w + in_ch * orig_h_w;
                dyn_var<int> inner_ker_idx = out_ch * ker_inch_w_h + in_ch * ker_w_h;

                int img_bounds_h[12];
                int ker_bounds_h[12];
                int img_bounds_w[12];
                int ker_bounds_w[12];
                get_bounds(img_bounds_h, ker_bounds_h, oh, wh, pad_h, stride[0], dilation[0], orig_ih, ih);
                get_bounds(img_bounds_w, ker_bounds_w, ow, ww, pad_w, stride[1], dilation[1], orig_iw, iw);
                for (static_var<int> r1 = 0; r1 < 6; r1 = r1 + 1) {
                    int h_lo = img_bounds_h[r1 * 2];
                    int h_hi = img_bounds_h[r1 * 2 + 1];
                    if (h_lo <= h_hi) {
                        builder::annotate("Comment: looping over the output");
                        for (dyn_var<int> h = h_lo; h < h_hi; h = h + 1) {
                            for (static_var<int> r2 = 0; r2 < 6; r2 = r2 + 1) {
                                int w_lo = img_bounds_w[r2 * 2];
                                int w_hi = img_bounds_w[r2 * 2 + 1];
                                if (w_lo <= w_hi) {
                                    for (dyn_var<int> w = w_lo; w < w_hi; w = w + 1) {
                                        out_idx =  bid * inch_oh_ow + out_ch * oh_times_ow + h * output.width + w;
                                        dyn_var<int> w_stride = w * stride[1];
                                        dyn_var<int> h_stride = h * stride[0];
                                        // if region = 2 the dilated kernel completely fits inside the orig image
                                        bool w_conditions = r1 != 2;
                                        bool h_conditions = r2 != 2;
                                        counter = counter + kernel_loops(inp_data, weight_data, output.data, h, w, w_stride, h_stride, out_idx, inner_img_idx, inner_ker_idx, ww, wh,
                                                        dilation, pad_h, pad_w, orig_ih, orig_iw, in_channels, out_channels, w_conditions, h_conditions);
                                    }
                                }
                            }
                        }
                    }
                }
                output.mult_cnt = counter; // should be safe to write in parallel since it's the same number for all iters
            }
        } 
    }
    output.mult_cnt = output.mult_cnt * batch_size * in_channels * out_channels;
    return output;
}

void update(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data,
            dyn_var<int>** curr_indices, int* stride, int* dilation, int orig_inch_h_w, int orig_h_w,
            int oh_times_ow, int inch_oh_ow, int ow, int ker_inch_w_h, int ker_w_h,
            int orig_iw, int ww, int pad_h, int pad_w) {
    dyn_var<int> bid = *(curr_indices[0]);
    dyn_var<int> in_ch = *(curr_indices[1]);
    dyn_var<int> out_ch = *(curr_indices[2]);
    dyn_var<int> h = *(curr_indices[3]);
    dyn_var<int> w = *(curr_indices[4]);
    dyn_var<int> i = *(curr_indices[5]);
    dyn_var<int> j = *(curr_indices[6]);
    dyn_var<int> out_idx = bid * inch_oh_ow + out_ch * oh_times_ow + h * ow + w;
    dyn_var<int> im_i = h * stride[0] + i * dilation[0];
    dyn_var<int> im_j = w * stride[1] + j * dilation[1];
    dyn_var<int> inner_img_idx = bid * orig_inch_h_w + in_ch * orig_h_w;
    dyn_var<int> inner_ker_idx = out_ch * ker_inch_w_h + in_ch * ker_w_h;
    dyn_var<int> img_val = input_data[inner_img_idx + (im_i - pad_h) * orig_iw + (im_j - pad_w)];
    dyn_var<int> weight_idx = inner_ker_idx + i * ww + j;
    output_data[out_idx] = output_data[out_idx] + img_val * weight_data[weight_idx];
}

void get_current_loop(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, 
                    dyn_var<int>** curr_indices,
                    Schedule s, LoopSchedule loop, int curr_loop, std::string annotation, 
                    int* img_bounds_h, int* img_bounds_w, int ww, int wh, int* stride, int* dilation, bool w_cond, bool h_cond, int pad_w, int pad_h, int orig_iw, int orig_ih,
                    int orig_inch_h_w, int orig_h_w, int oh_times_ow, int inch_oh_ow, int ow, int ker_inch_w_h, int ker_w_h, int oh, int r1, int r2) {
    if (curr_loop == s.n_loops - 1) {
        // innermost loop
        update(input_data, weight_data, output_data, curr_indices, stride, dilation, orig_inch_h_w, orig_h_w,
            oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, orig_iw, ww, pad_h, pad_w);
    } else {
        // go to the next loop
        get_loops(input_data, weight_data, output_data, 
                    curr_indices,  s, curr_loop + 1, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
    }
}

void get_loops(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, 
                dyn_var<int>** curr_indices, Schedule s, int curr_loop, int* img_bounds_h, int* img_bounds_w, int ww, 
                int wh, int* stride, int* dilation, bool w_cond, bool h_cond, int pad_w, int pad_h, int orig_iw, int orig_ih,
                int orig_inch_h_w, int orig_h_w, int oh_times_ow, int inch_oh_ow, int ow, int ker_inch_w_h, int ker_w_h, int oh, int r1, int r2) {
    assert(curr_loop < s.n_loops);
    LoopSchedule loop = s.loops[curr_loop];
    std::string loop_names[] = {
        "batches",
        "in channels",
        "out channels",
        "image rows",
        "image columns",
        "kernel rows",
        "kernel columns",
    };
    std::string annotation = "Comment: looping over " + loop_names[static_cast<int>(loop.type)];
    // add pragmas
    if (loop.parallel_collapse != 0) {
        annotation += " | #pragma omp parallel for collapse(" + std::to_string(loop.parallel_collapse) + ") ";
    } else if (loop.vectorized) {
        annotation += " | #pragma omp simd ";
    } else if (loop.unrolled) {
        annotation += ""; // TODO
    }
    int loop_type = static_cast<int>(loop.type);
    if (loop.type == LoopSchedule::loop_type::KW) {
        builder::annotate(annotation);
        for (dyn_var<int> j = 0; j < loop.bound; j = j + 1) {
            curr_indices[loop_type] = j.addr(); 
            if (h_cond && loop.after) {
                dyn_var<int> im_j = *(curr_indices[4]) * stride[1] + j * dilation[1];
                if (im_j < pad_w) continue;
                else if (im_j < orig_iw + pad_w) {
                     get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
                }
                else break;
            } else {
                 get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
            }
        }
    } else if (loop.type == LoopSchedule::loop_type::KH) {
        builder::annotate(annotation);
        for (dyn_var<int> i = 0; i < loop.bound; i = i + 1) {
            curr_indices[loop_type] = i.addr();
            if (w_cond && loop.after) {
                dyn_var<int> im_i = *(curr_indices[3]) * stride[0] + i * dilation[0];
                if (im_i < pad_h) continue;
                else if (im_i < orig_ih + pad_h) {
                     get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
                }
                else break;
            } else {
                 get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
            }
        }
    } else if (loop.type == LoopSchedule::loop_type::IH) {
        int h_lo = img_bounds_h[r1 * 2];
        int h_hi = img_bounds_h[r1 * 2 + 1];
        builder::annotate(annotation);
        for (dyn_var<int> i = h_lo; i < h_hi; i = i + 1) {
            curr_indices[loop_type] = i.addr();

            if (w_cond && loop.after) {
                dyn_var<int> im_i = i * stride[0] + *(curr_indices[5]) * dilation[0];
                if (im_i < pad_h) continue; 
                else if (im_i < orig_ih + pad_h) {
                    get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
                } else break;
            } else {
                get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
            }
        }
    } else if (loop.type == LoopSchedule::loop_type::IW) {
        int w_lo = img_bounds_w[r2 * 2];
        int w_hi = img_bounds_w[r2 * 2 + 1];
        builder::annotate(annotation);
        for (dyn_var<int> i = w_lo; i < w_hi; i = i + 1) {
            curr_indices[loop_type] = i.addr();
            if (h_cond && loop.after) {
                dyn_var<int> im_j = i * stride[1] + *(curr_indices[6]) * dilation[1];
                if (im_j < pad_w) continue; 
                else if (im_j < orig_iw + pad_w) {
                    get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
                } else break;
            } else {
                get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                    w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
            }
        }
    } else {
        builder::annotate(annotation);
        for (dyn_var<int> idx = 0; idx < loop.bound; idx = idx + 1) {
            curr_indices[loop_type] = idx.addr();
            get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, img_bounds_h, img_bounds_w, ww, wh, stride, dilation,
                        w_cond, h_cond, pad_w, pad_h, orig_iw, orig_ih, orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
        }
    }
}

ImageT<conv_t> static_conv2d_with_scheduling(dyn_var<conv_t*> inp_data, dyn_var<conv_t*> weight_data, int orig_iw, int orig_ih, int ww, int wh, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same, Schedule s) {

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
    static_var<int> oh_times_ow = oh * ow;
    static_var<int> inch_oh_ow = out_channels * oh_times_ow;

    static_var<int> orig_h_w = orig_ih * orig_iw;
    static_var<int> orig_inch_h_w = in_channels * orig_h_w;
    static_var<int> ker_w_h = ww * wh;
    static_var<int> ker_inch_w_h = in_channels * ker_w_h;

    ImageT<conv_t> output;
    output.height = oh;
    output.width = ow;
    output.in_channels = out_channels;
    output.batch_size = batch_size;
    static_var<int> size = ow * oh * batch_size * out_channels;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(conv_t));

    // getting image bounds with padding
    int img_bounds_h[12];
    int ker_bounds_h[12];
    int img_bounds_w[12];
    int ker_bounds_w[12];
    get_bounds(img_bounds_h, ker_bounds_h, oh, wh, pad_h, stride[0], dilation[0], orig_ih, ih);
    get_bounds(img_bounds_w, ker_bounds_w, ow, ww, pad_w, stride[1], dilation[1], orig_iw, iw);


    dyn_var<int>* curr_indices[7];
    for (static_var<int> r1 = 0; r1 < 6; r1 = r1 + 1) {
        int h_lo = img_bounds_h[r1 * 2];
        int h_hi = img_bounds_h[r1 * 2 + 1];
        if (h_lo <= h_hi) {
            for (static_var<int> r2 = 0; r2 < 6; r2 = r2 + 1) {
                int w_lo = img_bounds_w[r2 * 2];
                int w_hi = img_bounds_w[r2 * 2 + 1];
                if (w_lo <= w_hi) {
                    bool w_conditions = r1 != 2;
                    bool h_conditions = r2 != 2;
                    get_loops(inp_data, weight_data, output.data, curr_indices, s, 0, img_bounds_h, img_bounds_w, ww, 
                    wh, stride, dilation, w_conditions, h_conditions, pad_w, pad_h, orig_iw, orig_ih,
                    orig_inch_h_w, orig_h_w, oh_times_ow, inch_oh_ow, ow, ker_inch_w_h, ker_w_h, oh, r1, r2);
                }
            }
        } 
    }
    return output;
}




