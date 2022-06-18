#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"
#include "conv_functions/runtime.h"
#include "blocks/c_code_generator.h"
#include "conv_functions/schedule.h"

using builder::dyn_var;
using builder::static_var;
using conv::PaddingT;
using conv::ConvOptions;
using conv::ImageT;
using conv::Schedule;
using conv::LoopSchedule;

void get_bounds(int* img_bounds, int* ker_bounds, int out_size, int ker_size, int pad, int stride, int dilation, int orig_size, int im_size);

template <typename T>
void update_output(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, dyn_var<int> out_idx,
            dyn_var<int> im_i, dyn_var<int> im_j, dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, dyn_var<int> i, dyn_var<int> j,
            int orig_iw, int ww, int pad_h, int pad_w);

template <typename T>
dyn_var<int> kernel_w_loop(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, dyn_var<int> w_stride, 
    dyn_var<int> out_idx, dyn_var<int> im_i, dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, int ww, 
    dyn_var<int> i, int orig_iw, int pad_h, int pad_w, int dil, bool h_condition);

template <typename T>
dyn_var<int> kernel_loops(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data,
            dyn_var<int> h, dyn_var<int> w, dyn_var<int> w_stride, dyn_var<int> h_stride, dyn_var<int> out_idx, 
            dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, int ww, int wh,
            int* dilation, int pad_h, int pad_w,
            int orig_ih, int orig_iw, int in_channels, int out_channels, bool w_condition, bool h_condition);

template <typename T>
ImageT<T> static_conv2d_with_tiled_loops(dyn_var<T*> inp_data, dyn_var<T*> weight_data, int orig_iw, int orig_ih, int ww, int wh, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same);
template <typename T>
ImageT<T> static_conv2d_with_scheduling(dyn_var<T*> inp_data, dyn_var<T*> weight_data, int* orig_img_dims, int* ker_dims, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same, Schedule s, int ndims, int* out_dims, int* pad_dims, int* padded_img_dims);

template <typename T>
void get_loops(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
                dyn_var<int>** curr_indices, Schedule s, int curr_loop, int* stride, int* dilation, int* out_dims, 
                int* pad, int* orig, static_var<int>* r, int* img_bounds, int* ker_dims, int in_channels, int out_channels, int ndims);

template <typename T>
void get_current_loop(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
                    dyn_var<int>** curr_indices, 
                    Schedule s, LoopSchedule loop, int curr_loop, std::string annotation, 
                    int* stride, int* dilation, int* out_dims, int* pad, 
                    int* orig, static_var<int>* r, int* img_bounds, int* ker_dims, int in_channels, int out_channels, int ndims);

template <typename T>
void update(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data,
            dyn_var<int>** curr_indices, int* stride, int* dilation, int* out_dims, 
            int* orig_img_dims, int* ker_dims, int* pad, int in_channels, int out_channels, int ndims);

template <typename T>
void get_region_loops(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
    dyn_var<int>** curr_indices, Schedule s,
    int* stride, int* dilation, int* out_dims, int* pad, int* orig_img_dims, static_var<int>* regions, int* img_bounds, int* ker_dims,
    int in_channels, int out_channels, int curr_dim, int ndims); 