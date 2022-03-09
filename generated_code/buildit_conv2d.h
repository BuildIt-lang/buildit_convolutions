#include "runtime_functions.h"
#include "runtime_types.h"

conv_runtime::ImageT<int> buildit_conv2d (conv_runtime::ImageT<int> arg0, conv_runtime::KernelT<int> arg1, conv_runtime::ConvOptions arg2) {
  conv_runtime::ImageT<int> var3;
  if (((arg2.padding).is_same || ((arg2.padding).values[0] != 0)) || ((arg2.padding).values[1] != 0)) {
    conv_runtime::ImageT<int> var7;
    var7.batch_size = arg0.batch_size;
    if ((arg2.padding).is_same) {
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
    var7.data = conv_runtime::conv_malloc(((4 * var7.width) * var7.height) * var7.batch_size);
    int var10;
    int var11;
    // creating a padded image
    for (int var12 = 0; var12 < arg0.batch_size; var12 = var12 + 1) {
      for (int var13 = 0; var13 < var7.height; var13 = var13 + 1) {
        int var14 = 0;
        if (var14 < var7.width) {
          while (1) {
            var10 = (((var12 * var7.width) * var7.height) + (var13 * var7.width)) + var14;
            var11 = (((var12 * arg0.width) * arg0.height) + ((var13 - var8) * arg0.width)) + (var14 - var9);
            if ((((var13 < var8) || (var14 < var9)) || (var13 >= (arg0.height + var8))) || (var14 >= (arg0.width + var9))) {
              var7.data[var10] = 0;
            } else {
              var7.data[var10] = arg0.data[var11];
            }
            var14 = var14 + 1;
            if (var14 < var7.width) {
            } else {
              break;
            }
          }
        } 
      }
    }
    var3 = var7;
  } else {
    var3 = arg0;
  }
  conv_runtime::ImageT<int> var15;
  var15.height = (((var3.height - (arg2.dilation[0] * (arg1.height - 1))) - 1) / arg2.stride[0]) + 1;
  var15.width = (((var3.width - (arg2.dilation[1] * (arg1.width - 1))) - 1) / arg2.stride[1]) + 1;
  var15.batch_size = var3.batch_size;
  var15.data = conv_runtime::conv_malloc(4 * ((var15.width * var15.height) * var15.batch_size));
  int var17;
  int var18;
  // looping over batches
  for (int var19 = 0; var19 < var15.batch_size; var19 = var19 + 1) {
    // looping over the output
    for (int var20 = 0; var20 < var15.height; var20 = var20 + 1) {
      for (int var21 = 0; var21 < var15.width; var21 = var21 + 1) {
        var17 = (((var19 * var15.height) * var15.width) + (var20 * var15.width)) + var21;
        var15.data[var17] = 0;
        // looping over the kernel
        for (int var22 = 0; var22 < arg1.height; var22 = var22 + 1) {
          int var23 = 0;
          if (var23 < arg1.width) {
            while (1) {
              var18 = (((var19 * var3.width) * var3.height) + (((var20 * arg2.stride[0]) + (var22 * arg2.dilation[0])) * var3.width)) + ((var21 * arg2.stride[1]) + (var23 * arg2.dilation[1]));
              var15.data[var17] = var15.data[var17] + (var3.data[var18] * arg1.data[(var22 * arg1.width) + var23]);
              var23 = var23 + 1;
              if (var23 < arg1.width) {
              } else {
                break;
              }
            }
          } 
        }
      }
    }
  }
  return var15;
}

