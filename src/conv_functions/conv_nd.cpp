#include "conv_functions/conv_nd.h"

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
 * Updates the value at the current index of the image.
 */
template <typename T>
void update(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data,
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
 * If this is the innermost loop it calculates and updates the final image value at the current index.
 * Otherwise, recursively calls the next loop.
 * 
 */
template <typename T>
void get_next_loop(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
                    dyn_var<int>** curr_indices, 
                    Schedule s, LoopSchedule loop, int curr_loop, std::string annotation, 
                    int* stride, int* dilation, int* out_dims, int* pad, int* orig_img_dims, int* r, int* img_bounds, int* ker_dims, 
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

template <typename T>
void loop_body(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
                    dyn_var<int>** curr_indices, 
                    Schedule s, LoopSchedule loop, int curr_loop, std::string annotation, 
                    int* stride, int* dilation, int* out_dims, int* pad, int* orig, int* r, int* img_bounds, int* ker_dims, 
                    int in_channels, int out_channels, int ndims, int img_st_idx, int ker_st_idx) {
    if (r[loop.dim] != 2 && loop.after) {
        dyn_var<int> im_i = *(curr_indices[img_st_idx + loop.dim]) * stride[loop.dim] + *(curr_indices[ker_st_idx + loop.dim]) * dilation[loop.dim];
        if (im_i >= pad[loop.dim] && im_i < orig[loop.dim] + pad[loop.dim]) {
            get_next_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
            pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
        }
    } else {
        get_next_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
            pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
    }
}

/**
 * A recursive function that returns all the loops in the order given by the schedule.
 */
template <typename T>
void get_loops(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
                dyn_var<int>** curr_indices, Schedule s, int curr_loop, 
                int* stride, int* dilation, int* out_dims, int* pad, int* orig, int* r, int* img_bounds, int* ker_dims,
                int in_channels, int out_channels, int ndims) {
    assert(curr_loop < s.n_loops);
    LoopSchedule loop = s.loops[curr_loop];
    std::string loop_names[] = {
        "batches",
        "in channels",
        "out channels",
        "image",
        "kernel",
    };
    int loop_type = static_cast<int>(loop.type);
    std::string annotation = "Comment: looping over " + loop_names[loop_type];
    // add pragmas
    if (loop.parallel_collapse != 0) {
        annotation += " | #pragma omp parallel for collapse(" + std::to_string(loop.parallel_collapse) + ") ";
    } else if (loop.vectorized) {
        annotation += " | #pragma omp simd ";
    } else if (loop.unrolled) {
        // annotation += " | #pragma unroll";
    }
    int img_st_idx = 3;
    int ker_st_idx = img_st_idx + ndims;
    if (loop.type == LoopSchedule::loop_type::IMG && loop.tiled && !loop.last && r[loop.dim] != 2) {
        // if we are in one of the end regions and this is not the last tiled loop, skip the current loop (no tiling in end regions)
        get_next_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims, 
                pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
    
    } else if (loop.type == LoopSchedule::loop_type::IMG || loop.type == LoopSchedule::loop_type::KERNEL) {
        int h_lo = 0;
        int h_hi = loop.bound;
        int str = loop.stride;
        int st_idx = (loop.type == LoopSchedule::loop_type::IMG) ? img_st_idx : ker_st_idx;
        if (loop.type == LoopSchedule::loop_type::IMG) {
            // if it's an inner tiled loop and it's the center region use the adjusted bounds for tiling
            // only the outermost tiled loop uses the precomputed region bounds
            h_lo = (loop.tiled && !loop.first && r[loop.dim] == 2) ? 0 : (img_bounds + loop.dim * N_REGIONS * ndims)[r[loop.dim] * 2];
            h_hi = (loop.tiled && !loop.first && r[loop.dim] == 2) ? loop.bound : (img_bounds + loop.dim * N_REGIONS * ndims)[r[loop.dim] * 2 + 1];
            str = (r[loop.dim] == 2) ? loop.stride : 1;
        }
        if (loop.unrolled) {
            dyn_var<int> init_curr_idx = *(curr_indices[st_idx + loop.dim]);
            for (static_var<int> i = h_lo; i < h_hi; i = i + str) {
                dyn_var<int> total = init_curr_idx + i;
                curr_indices[st_idx + loop.dim] = total.addr();
                loop_body(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                        pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims, img_st_idx, ker_st_idx);
            }
        } else {
            builder::annotate(annotation);
            for (dyn_var<int> i = h_lo + *(curr_indices[st_idx + loop.dim]); i < *(curr_indices[st_idx + loop.dim]) + h_hi; i = i + str) {
                curr_indices[st_idx + loop.dim] = i.addr();
                // loop_body(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                //         pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims, img_st_idx, ker_st_idx);
                if (r[loop.dim] != 2 && loop.after) {
                    dyn_var<int> im_i = *(curr_indices[img_st_idx + loop.dim]) * stride[loop.dim] + *(curr_indices[ker_st_idx + loop.dim]) * dilation[loop.dim];
                    if (im_i < pad[loop.dim]) continue; 
                    else if (im_i < orig[loop.dim] + pad[loop.dim]) {
                        get_next_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                        pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
                    } else break;
                } else {
                    get_next_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                        pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
                }
            }
        }
    } else {
        if (loop.unrolled) {
            dyn_var<int> init_curr_idx = *(curr_indices[loop_type]);
            for (static_var<int> idx = 0; idx < loop.bound; idx = idx + loop.stride) {
                dyn_var<int> total = init_curr_idx + idx;
                curr_indices[loop_type] = total.addr();
                get_next_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                            pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
            }
        } else {
            builder::annotate(annotation);
            for (dyn_var<int> idx = *(curr_indices[loop_type]); idx < *(curr_indices[loop_type]) + loop.bound; idx = idx + loop.stride) {
                curr_indices[loop_type] = idx.addr();
                get_next_loop(input_data, weight_data, output_data, curr_indices, s, loop, curr_loop, annotation, stride, dilation, out_dims,
                            pad, orig, r, img_bounds, ker_dims, in_channels, out_channels, ndims);
            }
        }
    }
}

/**
 * Splits the image loops into regions based on whether the kernel 
 * is in the padded or the actual area of the image. These loops are static
 * and do not appear in the generated code.
 */
template <typename T>
void get_region_loops(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
    dyn_var<int>** curr_indices, Schedule s,
    int* stride, int* dilation, int* out_dims, int* pad, int* orig_img_dims, int* regions, int* img_bounds, int* ker_dims,
    int in_channels, int out_channels, int curr_dim, int ndims) {
        if (curr_dim == 0) {
            // the loops starting from here can be dynamic
            get_loops(input_data, weight_data, output_data, curr_indices, s, 0, stride, dilation, out_dims, 
                pad, orig_img_dims, regions, img_bounds, ker_dims, in_channels, out_channels, ndims);
        } else {
            for (static_var<int> ri = 0; ri < N_REGIONS; ri = ri + 1) {
                // get the bounds of the current region
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

template <typename T>
ImageT<T> conv_nd_main(dyn_var<T*> inp_data, dyn_var<T*> weight_data, int* orig_img_dims, int* ker_dims, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same, Schedule s, int ndims) {

    ImageT<T> output;
    output.dims = conv::runtime::int_malloc(ndims * (int)sizeof(int));
    static_var<int> size = batch_size * out_channels;

    std::unique_ptr<int> out_dims(new int[ndims]);
    std::unique_ptr<int> pad_dims(new int[ndims]);
    std::unique_ptr<int> padded_img_dims(new int[ndims]);

    std::unique_ptr<int> img_bounds(new int[ndims * ndims * N_REGIONS]);
    std::unique_ptr<int> ker_bounds(new int[ndims * ndims * N_REGIONS]);

    for (static_var<int> i = 0; i < ndims; i = i + 1) {
        if (padding_same == 1) {
            // pytorch does not support padding "same" with stride other than 1
            assert(stride[i] == 1);
            // calculate padding such that the output is the same shape as the input
            padded_img_dims.get()[i] = orig_img_dims[i] * stride[i] - stride[i] + dilation[i] * (ker_dims[i] - 1) + 1;
            pad_dims.get()[i] = (padded_img_dims.get()[i] - orig_img_dims[i]) / 2;
        } else {
            pad_dims.get()[i] = padding[i];
            padded_img_dims.get()[i] = orig_img_dims[i] + 2 * pad_dims.get()[i];
        }
        // dims of the output image
        out_dims.get()[i] = (padded_img_dims.get()[i] - dilation[i] * (ker_dims[i] - 1) - 1) / stride[i] + 1;
        output.dims[i] = out_dims.get()[i];

        size = size * out_dims.get()[i];

        // image bounds for padding
        get_bounds(img_bounds.get() + i * ndims * N_REGIONS, ker_bounds.get() + i * ndims * N_REGIONS, out_dims.get()[i], ker_dims[i], pad_dims.get()[i], stride[i], dilation[i], orig_img_dims[i], padded_img_dims.get()[i]);

    }

    output.in_channels = out_channels;
    output.batch_size = batch_size;
    output.data = conv::runtime::conv_calloc(size, (int)sizeof(T));

    int sz = 2 * ndims + 3;
    // an array to store the current indices for each loop type
    std::unique_ptr<dyn_var<int>*> curr_indices(new dyn_var<int>*[sz]);
    dyn_var<int> st = 0;
    for (static_var<int> i = 0; i < sz; i = i + 1) {
        curr_indices.get()[i] = st.addr();
    }

    std::unique_ptr<int> regions(new int[ndims]);
    get_region_loops(inp_data, weight_data, output.data, curr_indices.get(), s, stride, dilation, out_dims.get(), pad_dims.get(),
        orig_img_dims, regions.get(), img_bounds.get(), ker_dims, in_channels, out_channels, ndims, ndims);

    return output;
}

template ImageT<float> conv_nd_main(dyn_var<float*> inp_data, dyn_var<float*> weight_data, int* orig_img_dims, int* ker_dims, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same, Schedule s, int ndims);
template ImageT<int> conv_nd_main(dyn_var<int*> inp_data, dyn_var<int*> weight_data, int* orig_img_dims, int* ker_dims, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same, Schedule s, int ndims);


