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
ImageT<T> conv_nd_main(dyn_var<T*> inp_data, dyn_var<T*> weight_data, int* orig_img_dims, int* ker_dims, 
                    int batch_size, int in_channels, int out_channels, int* stride, int* dilation, 
                    int* padding, int padding_same, Schedule s, int ndims);

template <typename T>
void get_loops(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
                dyn_var<int>** curr_indices, Schedule s, int curr_loop, int* stride, int* dilation, int* out_dims, 
                int* pad, int* orig, int* r, int* img_bounds, int* ker_dims, int in_channels, int out_channels, int ndims);

template <typename T>
void get_next_loop(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
                    dyn_var<int>** curr_indices, 
                    Schedule s, LoopSchedule loop, int curr_loop, std::string annotation, 
                    int* stride, int* dilation, int* out_dims, int* pad, 
                    int* orig, int* r, int* img_bounds, int* ker_dims, int in_channels, int out_channels, int ndims);

template <typename T>
void update(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data,
            dyn_var<int>** curr_indices, int* stride, int* dilation, int* out_dims, 
            int* orig_img_dims, int* ker_dims, int* pad, int in_channels, int out_channels, int ndims);

template <typename T>
void get_region_loops(dyn_var<T*> input_data, dyn_var<T*> weight_data, dyn_var<T*> output_data, 
    dyn_var<int>** curr_indices, Schedule s,
    int* stride, int* dilation, int* out_dims, int* pad, int* orig_img_dims, int* regions, int* img_bounds, int* ker_dims,
    int in_channels, int out_channels, int curr_dim, int ndims); 