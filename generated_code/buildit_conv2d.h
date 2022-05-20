#include "runtime_functions.h"
#include "runtime_types.h"

#include <assert.h>

conv_runtime::ImageT<float> buildit_conv2d (conv_runtime::ImageT<float> arg0, conv_runtime::KernelT<float> arg1, conv_runtime::ConvOptions arg2) {
  assert(arg0.in_channels == arg1.in_channels);
  assert(arg1.height <= arg0.height);
  assert(arg1.width <= arg0.width);
  int var3 = (arg2.padding).values[0];
  int var4 = (arg2.padding).values[1];
  int var5;
  int var6;
  if ((arg2.padding).is_same) {
    assert((arg2.stride[0] == 1) && (arg2.stride[1] == 1));
    var5 = (((arg0.height * arg2.stride[0]) - arg2.stride[0]) + (arg2.dilation[0] * (arg1.height - 1))) + 1;
    var6 = (((arg0.width * arg2.stride[1]) - arg2.stride[1]) + (arg2.dilation[1] * (arg1.width - 1))) + 1;
    var3 = (var5 - arg0.height) / 2;
    var4 = (var6 - arg0.width) / 2;
  } else {
    var5 = arg0.height + (2 * var3);
    var6 = arg0.width + (2 * var4);
  }
  conv_runtime::ImageT<float> var7;
  var7.height = (((var5 - (arg2.dilation[0] * (arg1.height - 1))) - 1) / arg2.stride[0]) + 1;
  var7.width = (((var6 - (arg2.dilation[1] * (arg1.width - 1))) - 1) / arg2.stride[1]) + 1;
  var7.in_channels = arg1.out_channels;
  var7.batch_size = arg0.batch_size;
  var7.data = conv_runtime::conv_calloc(((var7.width * var7.height) * var7.batch_size) * var7.in_channels, 4);
  int var9;
  int var10;
  int var11;
  // looping over batches
  for (int var12 = 0; var12 < var7.batch_size; var12 = var12 + 1) {
    // looping over out channels
    for (int var13 = 0; var13 < arg1.out_channels; var13 = var13 + 1) {
      // looping over in channels
      for (int var14 = 0; var14 < arg0.in_channels; var14 = var14 + 1) {
        // looping over the output
        for (int var15 = 0; var15 < var7.height; var15 = var15 + 1) {
          for (int var16 = 0; var16 < var7.width; var16 = var16 + 1) {
            var9 = (((((var12 * var7.in_channels) * var7.height) * var7.width) + 
                ((var13 * var7.width) * var7.height)) + (var15 * var7.width)) + var16;
            // looping over the kernel
            for (int var17 = 0; var17 < arg1.height; var17 = var17 + 1) {
              for (int var18 = 0; var18 < arg1.width; var18 = var18 + 1) {
                int var19 = (var15 * arg2.stride[0]) + (var17 * arg2.dilation[0]);
                int var20 = (var16 * arg2.stride[1]) + (var18 * arg2.dilation[1]);
                int var21;
                if ((((var19 < var3) || (var20 < var4)) || (var19 >= (arg0.height + var3))) || (var20 >= (arg0.width + var4))) {
                  var21 = 0;
                } else {
                  var21 = arg0.data[(((((var12 * arg0.in_channels) * arg0.width) * arg0.height) + 
                      ((var14 * arg0.width) * arg0.height)) + ((var19 - var3) * arg0.width)) + (var20 - var4)];
                }
                var11 = (((((var13 * arg1.in_channels) * arg1.width) * arg1.height) + 
                    ((var14 * arg1.width) * arg1.height)) + (var17 * arg1.width)) + var18;
                var7.data[var9] = var7.data[var9] + (var21 * arg1.data[var11]);
              }
            }
          }
        }
      }
    }
  }
  return var7;
}

