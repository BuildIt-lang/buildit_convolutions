#ifndef CONV_TYPES_H
#define CONV_TYPES_H
#include "builder/dyn_var.h"
#include "builder/builder.h"

using builder::dyn_var;
using builder::as_member_of;

namespace conv {

#define TENSOR_T_NAME "conv_runtime::TensorT<int>"
extern const char tensor_t_name[sizeof(TENSOR_T_NAME)];

#define CONVOPTIONS_T_NAME "conv_runtime::ConvOptions"
extern const char convoptions_t_name[sizeof(CONVOPTIONS_T_NAME)];

// this is a 2D tensor for now
struct TensorT: public dyn_var<builder::name<tensor_t_name>> {
    typedef dyn_var<builder::name<tensor_t_name>> super;
    using super_name = builder::name<tensor_t_name>;
    using super::dyn_var;
    using super::operator=;
    TensorT(const TensorT &t): super((builder::builder)t) {}
    builder::builder operator= (const TensorT &t) {
		return (*this) = (builder::builder)t;
	}
	TensorT* addr(void) {
		return this;
	}
    dyn_var<int> width = as_member_of(this, "width");
    dyn_var<int> height = as_member_of(this, "height");
    dyn_var<int*> data = as_member_of(this, "data"); // array of shape width*height
    dyn_var<void(void)> print = as_member_of(this, "print");
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
    ConvOptions* assr(void) {
        return this;
    }
    // 2D arrays for 2D convolution (height, width)
    dyn_var<int*> stride = as_member_of(this, "stride");
    dyn_var<int*> padding = as_member_of(this, "padding");
    dyn_var<int*> dilation = as_member_of(this, "dilation");
    dyn_var<int> groups = as_member_of(this, "groups");
};


}

#endif