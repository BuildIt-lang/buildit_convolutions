#include <assert.h>

conv_runtime::ImageT<int> f1 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  var4 = 10 + (2 * var2);
  var5 = 10 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 2) - 1) / 1) + 1;
  var6.width = (((var5 - 2) - 1) / 1) + 1;
  var6.in_channels = 1;
  var6.batch_size = 10;
  var6.data = conv_runtime::conv_calloc(((var6.width * var6.height) * var6.batch_size) * var6.in_channels, 4);
  int var8;
  int var9;
  // looping over batches
  for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
    // looping over out channels
    for (int var11 = 0; var11 < 1; var11 = var11 + 1) {
      // looping over in channels
      for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
        // looping over the output
        for (int var13 = 0; var13 < var6.height; var13 = var13 + 1) {
          for (int var14 = 0; var14 < var6.width; var14 = var14 + 1) {
            var8 = (((((var10 * 10) * var6.height) * var6.width) + ((var11 * var6.width) * var6.height)) + (var13 * var6.width)) + var14;
            // looping over the kernel
            for (int var15 = 0; var15 < 3; var15 = var15 + 1) {
              for (int var16 = 0; var16 < 3; var16 = var16 + 1) {
                int var17 = (var13 * 1) + (var15 * 1);
                int var18 = (var14 * 1) + (var16 * 1);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (10 + var2))) || (var18 >= (10 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 10) * 10) * 10) + ((var12 * 10) * 10)) + ((var17 - var2) * 10)) + (var18 - var3)];
                }
                var9 = (((((var11 * 10) * 3) * 3) + ((var12 * 3) * 3)) + (var15 * 3)) + var16;
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



conv_runtime::ImageT<int> f2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  var4 = 100 + (2 * var2);
  var5 = 100 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 9) - 1) / 1) + 1;
  var6.width = (((var5 - 9) - 1) / 1) + 1;
  var6.in_channels = 1;
  var6.batch_size = 10;
  var6.data = conv_runtime::conv_calloc(((var6.width * var6.height) * var6.batch_size) * var6.in_channels, 4);
  int var8;
  int var9;
  // looping over batches
  for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
    // looping over out channels
    for (int var11 = 0; var11 < 1; var11 = var11 + 1) {
      // looping over in channels
      for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
        // looping over the output
        for (int var13 = 0; var13 < var6.height; var13 = var13 + 1) {
          for (int var14 = 0; var14 < var6.width; var14 = var14 + 1) {
            var8 = (((((var10 * 10) * var6.height) * var6.width) + ((var11 * var6.width) * var6.height)) + (var13 * var6.width)) + var14;
            // looping over the kernel
            for (int var15 = 0; var15 < 10; var15 = var15 + 1) {
              for (int var16 = 0; var16 < 10; var16 = var16 + 1) {
                int var17 = (var13 * 1) + (var15 * 1);
                int var18 = (var14 * 1) + (var16 * 1);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (100 + var2))) || (var18 >= (100 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 10) * 100) * 100) + ((var12 * 100) * 100)) + ((var17 - var2) * 100)) + (var18 - var3)];
                }
                var9 = (((((var11 * 10) * 10) * 10) + ((var12 * 10) * 10)) + (var15 * 10)) + var16;
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



conv_runtime::ImageT<int> f3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  var4 = 100 + (2 * var2);
  var5 = 1000 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 9) - 1) / 1) + 1;
  var6.width = (((var5 - 9) - 1) / 1) + 1;
  var6.in_channels = 1;
  var6.batch_size = 10;
  var6.data = conv_runtime::conv_calloc(((var6.width * var6.height) * var6.batch_size) * var6.in_channels, 4);
  int var8;
  int var9;
  // looping over batches
  for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
    // looping over out channels
    for (int var11 = 0; var11 < 1; var11 = var11 + 1) {
      // looping over in channels
      for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
        // looping over the output
        for (int var13 = 0; var13 < var6.height; var13 = var13 + 1) {
          for (int var14 = 0; var14 < var6.width; var14 = var14 + 1) {
            var8 = (((((var10 * 10) * var6.height) * var6.width) + ((var11 * var6.width) * var6.height)) + (var13 * var6.width)) + var14;
            // looping over the kernel
            for (int var15 = 0; var15 < 10; var15 = var15 + 1) {
              for (int var16 = 0; var16 < 10; var16 = var16 + 1) {
                int var17 = (var13 * 1) + (var15 * 1);
                int var18 = (var14 * 1) + (var16 * 1);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (100 + var2))) || (var18 >= (1000 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 10) * 1000) * 100) + ((var12 * 1000) * 100)) + ((var17 - var2) * 1000)) + (var18 - var3)];
                }
                var9 = (((((var11 * 10) * 10) * 10) + ((var12 * 10) * 10)) + (var15 * 10)) + var16;
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



