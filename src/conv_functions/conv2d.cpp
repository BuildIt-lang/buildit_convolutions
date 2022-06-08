#include "conv_functions/conv2d.h"

int N_REGIONS = 6;

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
 * Splits the image loops based on the position of the kernel wrt the image.
 * Currently works only for padding value 0.
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
    output.dims = conv::runtime::int_malloc(2 * (int)sizeof(int));
    output.dims[0] = oh;
    output.dims[1] = ow;

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
                                        out_idx =  bid * inch_oh_ow + out_ch * oh_times_ow + h * ow + w;
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

/**
 * Updates the value at the current index of the image.
 */

void update(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data,
            dyn_var<int>** curr_indices, int* stride, int* dilation, int* out_dims, 
            int* orig_img_dims, int* ker_dims, int* pad, int in_channels, int out_channels, int ndims) {
    dyn_var<int> bid = *(curr_indices[0]);
    dyn_var<int> in_ch = *(curr_indices[1]);
    dyn_var<int> out_ch = *(curr_indices[2]);
    int img_st_idx = 3;
    int ker_st_idx = img_st_idx + ndims;
    dyn_var<int> out_idx = 0; // index into the result image
    dyn_var<int> img_idx = 0; // index into the input image
    dyn_var<int> ker_idx = 0; // index into the kernel
    static_var<int> out_depth = 1;
    static_var<int> ker_depth = 1;
    static_var<int> img_depth = 1;
    for (static_var<int> d = ndims - 1; d >= 0; d = d - 1) {
        out_idx = out_idx + *(curr_indices[img_st_idx + d]) * out_depth;
        out_depth = out_depth * out_dims[d];

        ker_idx = ker_idx + *(curr_indices[ker_st_idx + d]) * ker_depth;
        ker_depth = ker_depth * ker_dims[d];

        img_idx = img_idx + (*(curr_indices[img_st_idx + d]) * stride[d] + *(curr_indices[ker_st_idx + d]) * dilation[d] - pad[d]) * img_depth;
        img_depth = img_depth * orig_img_dims[d];
    }
    out_idx = out_idx + out_ch * out_depth + bid * out_channels * out_depth;
    ker_idx = ker_idx + in_ch * ker_depth + out_ch * in_channels * ker_depth;
    img_idx = img_idx + in_ch * img_depth + bid * in_channels * img_depth;
    output_data[out_idx] = output_data[out_idx] + input_data[img_idx] * weight_data[ker_idx];
}

/**
 * If this is the innermost loop, it calculates and updates the final image value at the current index.
 * Otherwise, recursively calls the next loop.
 * 
 */
void get_current_loop(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, 
                    dyn_var<int>** curr_indices,
                    Schedule s, LoopSchedule loop, int curr_loop, std::string annotation, 
                    int* stride, int* dilation, int* out_dims, int* pad, int* orig_img_dims, static_var<int>* r, int* img_bounds, int* ker_dims, 
                    int in_channels, int out_channels, int ndims) {
    if (curr_loop == s.n_loops - 1) {
        // innermost loop
        update(input_data, weight_data, output_data, curr_indices, stride, dilation, out_dims, orig_img_dims, ker_dims, pad, in_channels, out_channels, ndims);
    } else {
        // go to the next loop
        get_loops(input_data, weight_data, output_data, 
                    curr_indices,  s, curr_loop + 1, stride, dilation, out_dims, 
                    pad, orig_img_dims, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
    }
}


/**
 * A recursive function that returns all the loops in the order given by the schedule.
 */
void get_loops(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, 
                dyn_var<int>** curr_indices, Schedule s, int curr_loop, 
                int* stride, int* dilation, int* out_dims, int* pad, int* orig, static_var<int>* r, int* img_bounds, int* ker_dims,
                int in_channels, int out_channels, int ndims) {
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
    int img_st_idx = 3;
    int ker_st_idx = img_st_idx + ndims;
    if (loop.type == LoopSchedule::loop_type::KERNEL) {
        builder::annotate(annotation);
        for (dyn_var<int> j = 0; j < loop.bound; j = j + loop.stride) {
            dyn_var<int> total = *(curr_indices[loop_type + loop.dim]) + j;
            curr_indices[loop_type + loop.dim] = total.addr(); 
            if (r[loop.dim] != 2 && loop.after) {
                dyn_var<int> im_idx = *(curr_indices[img_st_idx + loop.dim]) * stride[loop.dim] + *(curr_indices[loop_type + loop.dim]) * dilation[loop.dim];
                if (im_idx < pad[loop.dim]) continue;
                else if (im_idx < orig[loop.dim] + pad[loop.dim]) {
                     get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                     pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
                }
                else break;
            } else {
                 get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims, 
                    pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
            }
        }
    } else if (loop.type == LoopSchedule::loop_type::IMG) {
        if (loop.tiled && !loop.last && r[loop.dim] != 2) {
            // if we are in one of the end regions and this is not the last tiled loop, skip the current loop (no tiling in end regions)
            get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims, 
                    pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
        
        } else {
            // if it's the innermost tiled loop and it's the center region use the adjusted bounds for tiling
            int h_lo = (loop.tiled && !loop.first && r[loop.dim] == 2) ? 0 : (img_bounds + loop.dim * N_REGIONS * ndims)[r[loop.dim] * 2];
            int h_hi = (loop.tiled && !loop.first && r[loop.dim] == 2) ? loop.bound : (img_bounds + loop.dim * N_REGIONS * ndims)[r[loop.dim] * 2 + 1];
            int str = (r[loop.dim] == 2) ? loop.stride : 1;
            builder::annotate(annotation);
            for (dyn_var<int> i = h_lo; i < h_hi; i = i + str) {
                dyn_var<int> total = *(curr_indices[loop_type + loop.dim]) + i;
                curr_indices[loop_type + loop.dim] = total.addr();

                if (r[loop.dim] != 2 && loop.after) {
                    dyn_var<int> im_i = *(curr_indices[loop_type + loop.dim]) * stride[loop.dim] + *(curr_indices[ker_st_idx + loop.dim]) * dilation[loop.dim];
                    if (im_i < pad[loop.dim]) continue; 
                    else if (im_i < orig[loop.dim] + pad[loop.dim]) {
                        get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                        pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
                    } else break;
                } else {
                    get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                        pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
                }
            }
        }
    } else {
        builder::annotate(annotation);
        for (dyn_var<int> idx = 0; idx < loop.bound; idx = idx + loop.stride) {
            dyn_var<int> total = *(curr_indices[loop_type]) + idx;
            curr_indices[loop_type] = total.addr();
            get_current_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                        pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
        }
    }
}

void get_region_loops(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, 
    dyn_var<int>** curr_indices, Schedule s,
    int* stride, int* dilation, int* out_dims, int* pad, int* orig_img_dims, static_var<int>* regions, int* img_bounds, int* ker_dims,
    int in_channels, int out_channels, int curr_dim, int ndims) {
        if (curr_dim == 0) {
            get_loops(input_data, weight_data, output_data, curr_indices, s, 0, stride, dilation, out_dims, 
                pad, orig_img_dims, regions, img_bounds, ker_dims, in_channels, out_channels, ndims);
        } else {
            for (static_var<int> ri = 0; ri < N_REGIONS; ri = ri + 1) {
                int lo = (img_bounds + (curr_dim - 1) * ndims * N_REGIONS)[ri * 2];
                int hi = (img_bounds + (curr_dim - 1) * ndims * N_REGIONS)[ri * 2 + 1];
                if (lo <= hi) {
                    regions[curr_dim - 1] = ri;
                    get_region_loops(input_data, weight_data, output_data, curr_indices, s, stride, dilation, out_dims,
                    pad, orig_img_dims, regions, img_bounds, ker_dims, in_channels, out_channels, curr_dim - 1, ndims);
                } 
            }
        }
    }

ImageT<conv_t> static_conv2d_with_scheduling(dyn_var<conv_t*> inp_data, dyn_var<conv_t*> weight_data, int* orig_img_dims, int* ker_dims, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same, Schedule s, int ndims, int* out_dims, int* pad_dims, int* padded_img_dims) {

    ImageT<conv_t> output;
    output.dims = conv::runtime::int_malloc(ndims * (int)sizeof(int));
    static_var<int> size = batch_size * out_channels;

    int* img_bounds = (int*)malloc(ndims * ndims * N_REGIONS * (int)sizeof(int));
    int* ker_bounds = (int*)malloc(ndims * ndims * N_REGIONS * (int)sizeof(int));

    for (static_var<int> i = 0; i < ndims; i = i + 1) {
        if (padding_same == 1) {
            // torch does not support padding "same" with stride other than 1
            conv::runtime::conv_assert(stride[i] == 1);
            // calculate padding such that the output is the same shape as the input
            padded_img_dims[i] = orig_img_dims[i] * stride[i] - stride[i] + dilation[i] * (ker_dims[i] - 1) + 1;
            pad_dims[i] = (padded_img_dims[i] - orig_img_dims[i]) / 2;
        } else {
            pad_dims[i] = padding[i];
            padded_img_dims[i] = orig_img_dims[i] + 2 * pad_dims[i];
        }
        // dims of the output image
        out_dims[i] = (padded_img_dims[i] - dilation[i] * (ker_dims[i] - 1) - 1) / stride[i] + 1;
        output.dims[i] = out_dims[i];

        size = size * out_dims[i];

        // image bounds for padding
        get_bounds(img_bounds + i * ndims * N_REGIONS, ker_bounds + i * ndims * N_REGIONS, out_dims[i], ker_dims[i], pad_dims[i], stride[i], dilation[i], orig_img_dims[i], padded_img_dims[i]);

    }

    output.in_channels = out_channels;
    output.batch_size = batch_size;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(conv_t));

    // TODO: generalize this
    dyn_var<int>* curr_indices[20];
    dyn_var<int> st = 0;
    for (static_var<int> i = 0; i < 20; i = i + 1) {
        curr_indices[i] = st.addr();
    }
    
    static_var<int>* regions = (static_var<int>*)malloc(ndims * (int)sizeof(int)); // indices of the regions
    get_region_loops(inp_data, weight_data, output.data, curr_indices, s, stride, dilation, out_dims, pad_dims,
        orig_img_dims, regions, img_bounds, ker_dims, in_channels, out_channels, ndims, ndims);

    free(img_bounds);
    free(ker_bounds);
    free(regions);
    return output;
}


