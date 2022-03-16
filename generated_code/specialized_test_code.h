#include <assert.h>

conv_runtime::ImageT<int> conv2d_default_im5x5_w3x3 (conv_runtime::ImageT<int> arg0, conv_runtime::KernelT<int> arg1) {
  assert(arg0.in_channels == arg1.in_channels);
  assert(arg1.height <= arg0.height);
  assert(arg1.width <= arg0.width);
  conv_runtime::ImageT<int> var2;
  var2 = arg0;
  conv_runtime::ImageT<int> var3;
  var3.height = (((var2.height - (1 * (arg1.height - 1))) - 1) / 1) + 1;
  var3.width = (((var2.width - (1 * (arg1.width - 1))) - 1) / 1) + 1;
  var3.in_channels = arg1.out_channels;
  var3.batch_size = var2.batch_size;
  var3.data = conv_runtime::conv_calloc(((var3.width * var3.height) * var3.batch_size) * var3.in_channels, 4);
  int var5;
  int var6;
  int var7;
  // looping over batches
  for (int var8 = 0; var8 < var3.batch_size; var8 = var8 + 1) {
    // looping over out channels
    for (int var9 = 0; var9 < arg1.out_channels; var9 = var9 + 1) {
      // looping over in channels
      for (int var10 = 0; var10 < var2.in_channels; var10 = var10 + 1) {
        // looping over the output
        for (int var11 = 0; var11 < var3.height; var11 = var11 + 1) {
          for (int var12 = 0; var12 < var3.width; var12 = var12 + 1) {
            var5 = (((((var8 * var3.in_channels) * var3.height) * var3.width) + ((var9 * var3.width) * var3.height)) + (var11 * var3.width)) + var12;
            // looping over the kernel
            for (int var13 = 0; var13 < arg1.height; var13 = var13 + 1) {
              for (int var14 = 0; var14 < arg1.width; var14 = var14 + 1) {
                var6 = (((((var8 * var2.in_channels) * var2.width) * var2.height) + ((var10 * var2.width) * var2.height)) + (((var11 * 1) + (var13 * 1)) * var2.width)) + ((var12 * 1) + (var14 * 1));
                var7 = (((((var9 * arg1.in_channels) * arg1.width) * arg1.height) + ((var10 * arg1.width) * arg1.height)) + (var13 * arg1.width)) + var14;
                var3.data[var5] = var3.data[var5] + (var2.data[var6] * arg1.data[var7]);
              }
            }
          }
        }
      }
    }
  }
  return var3;
}



