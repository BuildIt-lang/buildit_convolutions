#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "conv_functions/conv_types.h"

using builder::dyn_var;
using builder::static_var;
using conv::ConvOptions;
using conv::PaddingT;
using conv::ImageT;
using conv::KernelT;

typedef float conv_t;

ImageT<conv_t> dyn_conv2d(ImageT<conv_t> input, KernelT<conv_t> weight, ConvOptions options);

ImageT<conv_t> static_conv2d(dyn_var<conv_t*> inp_data, dyn_var<conv_t*> weight_data, int iw, int ih, int ww, int wh, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same);

void get_bounds(int* img_bounds, int* ker_bounds, int out_size, int ker_size, int pad, int stride, int dilation, int orig_size, int im_size);

void update_output(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, dyn_var<int> out_idx,
            dyn_var<int> im_i, dyn_var<int> im_j, dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, dyn_var<int> i, dyn_var<int> j,
            int orig_iw, int ww, int pad_h, int pad_w);

dyn_var<int> kernel_w_loop(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data, dyn_var<int> w_stride, 
    dyn_var<int> out_idx, dyn_var<int> im_i, dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, int ww, 
    dyn_var<int> i, int orig_iw, int pad_h, int pad_w, int dil, bool h_condition);

dyn_var<int> kernel_loops(dyn_var<conv_t*> input_data, dyn_var<conv_t*> weight_data, dyn_var<conv_t*> output_data,
            dyn_var<int> h, dyn_var<int> w, dyn_var<int> w_stride, dyn_var<int> h_stride, dyn_var<int> out_idx, 
            dyn_var<int> inner_img_idx, dyn_var<int> inner_ker_idx, int ww, int wh,
            int* dilation, int pad_h, int pad_w,
            int orig_ih, int orig_iw, int in_channels, int out_channels, bool w_condition, bool h_condition);

ImageT<conv_t> static_conv2d_with_tiled_loops(dyn_var<conv_t*> inp_data, dyn_var<conv_t*> weight_data, int orig_iw, int orig_ih, int ww, int wh, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same);
