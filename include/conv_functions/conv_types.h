#ifndef CONV_TYPES_H
#define CONV_TYPES_H
#include "builder/dyn_var.h"
#include "builder/builder.h"

namespace conv {
using builder::dyn_var;
using builder::as_member_of;


#define CONVOPTIONS_T_NAME "conv_runtime::ConvOptions"
extern const char convoptions_t_name[sizeof(CONVOPTIONS_T_NAME)];

#define PADDING_T_NAME "conv_runtime::PaddingT"
extern const char padding_t_name[sizeof(PADDING_T_NAME)];

#define IMAGE_T_NAME "conv_runtime::ImageT"
extern const char image_t_name[sizeof(IMAGE_T_NAME)];

template <typename T>
struct ImageT: public dyn_var<builder::name<image_t_name, T>> {
    typedef dyn_var<builder::name<image_t_name, T>> super;

    using super::super;
    using super::operator=;
    ImageT(const ImageT &t): super((builder::builder)t) {}
    builder::builder operator= (const ImageT &t) {
		return (*this) = (builder::builder)t;
	}

    dyn_var<int> batch_size = as_member_of(this, "batch_size");
    dyn_var<int> in_channels = as_member_of(this, "in_channels");
    dyn_var<int> ndims = as_member_of(this, "ndims");
    dyn_var<int*> dims = as_member_of(this, "dims");
    dyn_var<T*> data = as_member_of(this, "data");
    dyn_var<void(void)> print = as_member_of(this, "print");
    dyn_var<int> mult_cnt = as_member_of(this, "mult_cnt");
};


struct PaddingT: public dyn_var<builder::name<padding_t_name>> {
    typedef dyn_var<builder::name<padding_t_name>> super;
    using super_name = builder::name<padding_t_name>;
    using super::dyn_var;
    using super::operator=;
    PaddingT(const PaddingT &t): dyn_var<super_name>((builder::builder)t) {}
    builder::builder operator= (const PaddingT &t) {
        return (*this) = (builder::builder)t;
    }

    dyn_var<bool> is_same = as_member_of(this, "is_same");
    dyn_var<int*> values = as_member_of(this, "values"); // 2D array
};


struct ConvOptions: public dyn_var<builder::name<convoptions_t_name>> {
    typedef dyn_var<builder::name<convoptions_t_name>> super;
    using super_name = builder::name<convoptions_t_name>;
    using super::dyn_var;
    using super::operator=;
    ConvOptions(const ConvOptions &t): dyn_var<super_name>((builder::builder)t) {}
    builder::builder operator= (const ConvOptions &t) {
        return (*this) = (builder::builder)t;
    }

    // 2D arrays for 2D convolution (height, width)
    dyn_var<int*> stride = as_member_of(this, "stride");
    PaddingT padding = as_member_of(this, "padding");
    dyn_var<int*> dilation = as_member_of(this, "dilation");
    dyn_var<int> groups = as_member_of(this, "groups");
};


}

#endif
