#include "runtime_functions.h"
#include "runtime_types.h"

#include <assert.h>

conv_runtime::ImageT<int> buildit_conv2d (conv_runtime::ImageT<int> arg0, conv_runtime::KernelT<int> arg1, conv_runtime::ConvOptions arg2) {
  assert(arg0.in_channels == arg1.in_channels);
  assert(arg1.height <= arg0.height);
  assert(arg1.width <= arg0.width);
  conv_runtime::ImageT<int> var3;
  if (((arg2.padding).is_same || ((arg2.padding).values[0] != 0)) || ((arg2.padding).values[1] != 0)) {
    conv_runtime::ImageT<int> var7;
    var7.batch_size = arg0.batch_size;
    var7.in_channels = arg0.in_channels;
    if ((arg2.padding).is_same) {
      assert((arg2.stride[0] == 1) && (arg2.stride[1] == 1));
      var7.height = (((arg0.height * arg2.stride[0]) - arg2.stride[0]) + (arg2.dilation[0] * (arg1.height - 1))) + 1;
      var7.width = (((arg0.width * arg2.stride[1]) - arg2.stride[1]) + (arg2.dilation[1] * (arg1.width - 1))) + 1;
      (arg2.padding).values[0] = (var7.height - arg0.height) / 2;
      (arg2.padding).values[1] = (var7.width - arg0.width) / 2;
    } else {
      var7.height = arg0.height + (2 * (arg2.padding).values[0]);
      var7.width = arg0.width + (2 * (arg2.padding).values[1]);
    }
    int var8 = (arg2.padding).values[0];
    int var9 = (arg2.padding).values[1];
    var7.data = conv_runtime::conv_malloc((((4 * var7.width) * var7.height) * var7.batch_size) * var7.in_channels);
    int var10;
    int var11;
    // creating a padded image
    for (int var12 = 0; var12 < arg0.batch_size; var12 = var12 + 1) {
      for (int var13 = 0; var13 < arg0.in_channels; var13 = var13 + 1) {
        for (int var14 = 0; var14 < var7.height; var14 = var14 + 1) {
          for (int var15 = 0; var15 < var7.width; var15 = var15 + 1) {
            var10 = (((((var12 * var7.in_channels) * var7.width) * var7.height) + ((var13 * var7.width) * var7.height)) + (var14 * var7.width)) + var15;
            var11 = (((((var12 * arg0.in_channels) * arg0.width) * arg0.height) + ((var13 * arg0.width) * arg0.height)) + ((var14 - var8) * arg0.width)) + (var15 - var9);
            if ((((var14 < var8) || (var15 < var9)) || (var14 >= (arg0.height + var8))) || (var15 >= (arg0.width + var9))) {
              var7.data[var10] = 0;
            } else {
              var7.data[var10] = arg0.data[var11];
            }
          }
        }
      }
    }
    var3 = var7;
  } else {
    var3 = arg0;
  }
  conv_runtime::ImageT<int> var16;
  var16.height = (((var3.height - (arg2.dilation[0] * (arg1.height - 1))) - 1) / arg2.stride[0]) + 1;
  var16.width = (((var3.width - (arg2.dilation[1] * (arg1.width - 1))) - 1) / arg2.stride[1]) + 1;
  var16.in_channels = arg1.out_channels;
  var16.batch_size = var3.batch_size;
  var16.data = conv_runtime::conv_calloc(((var16.width * var16.height) * var16.batch_size) * var16.in_channels, 4);
  int var18;
  int var19;
  int var20;
  // looping over batches
  for (int var21 = 0; var21 < var16.batch_size; var21 = var21 + 1) {
    // looping over out channels
    for (int var22 = 0; var22 < arg1.out_channels; var22 = var22 + 1) {
      // looping over in channels
      for (int var23 = 0; var23 < var3.in_channels; var23 = var23 + 1) {
        // looping over the output
        for (int var24 = 0; var24 < var16.height; var24 = var24 + 1) {
          for (int var25 = 0; var25 < var16.width; var25 = var25 + 1) {
            var18 = (((((var21 * var16.in_channels) * var16.height) * var16.width) + ((var22 * var16.width) * var16.height)) + (var24 * var16.width)) + var25;
            // looping over the kernel
            for (int var26 = 0; var26 < arg1.height; var26 = var26 + 1) {
              for (int var27 = 0; var27 < arg1.width; var27 = var27 + 1) {
                var19 = (((((var21 * var3.in_channels) * var3.width) * var3.height) + ((var23 * var3.width) * var3.height)) + (((var24 * arg2.stride[0]) + (var26 * arg2.dilation[0])) * var3.width)) + ((var25 * arg2.stride[1]) + (var27 * arg2.dilation[1]));
                var20 = (((((var22 * arg1.in_channels) * arg1.width) * arg1.height) + ((var23 * arg1.width) * arg1.height)) + (var26 * arg1.width)) + var27;
                var16.data[var18] = var16.data[var18] + (var3.data[var19] * arg1.data[var20]);
              }
            }
          }
        }
      }
    }
  }
  return var16;
}

