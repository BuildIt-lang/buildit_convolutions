#include <assert.h>

conv_runtime::ImageT<int> conv2d_default_im5x5_w3x3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  var4 = 5 + (2 * var2);
  var5 = 5 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 2) - 1) / 1) + 1;
  var6.width = (((var5 - 2) - 1) / 1) + 1;
  var6.in_channels = 1;
  var6.batch_size = 1;
  var6.data = conv_runtime::conv_calloc(((var6.width * var6.height) * var6.batch_size) * var6.in_channels, 4);
  int var8;
  int var9;
  // looping over batches
  for (int var10 = 0; var10 < 1; var10 = var10 + 1) {
    // looping over out channels
    for (int var11 = 0; var11 < 1; var11 = var11 + 1) {
      // looping over in channels
      for (int var12 = 0; var12 < 1; var12 = var12 + 1) {
        // looping over the output
        for (int var13 = 0; var13 < var6.height; var13 = var13 + 1) {
          for (int var14 = 0; var14 < var6.width; var14 = var14 + 1) {
            var8 = (((((var10 * 1) * var6.height) * var6.width) + ((var11 * var6.width) * var6.height)) + (var13 * var6.width)) + var14;
            // looping over the kernel
            for (int var15 = 0; var15 < 3; var15 = var15 + 1) {
              for (int var16 = 0; var16 < 3; var16 = var16 + 1) {
                int var17 = (var13 * 1) + (var15 * 1);
                int var18 = (var14 * 1) + (var16 * 1);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (5 + var2))) || (var18 >= (5 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 5) * 5) + ((var12 * 5) * 5)) + ((var17 - var2) * 5)) + (var18 - var3)];
                }
                var9 = (((((var11 * 1) * 3) * 3) + ((var12 * 3) * 3)) + (var15 * 3)) + var16;
                var6.data[var8] = var6.data[var8] + (var19 * arg1[var9]);
              }
            }
          }
        }
      }
    }
  }
  return var6;
}



