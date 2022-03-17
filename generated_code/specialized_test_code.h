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



conv_runtime::ImageT<int> conv2d_stride2x1_im8x10_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  var4 = 8 + (2 * var2);
  var5 = 10 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 2) - 1) / 2) + 1;
  var6.width = (((var5 - 1) - 1) / 1) + 1;
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
              for (int var16 = 0; var16 < 2; var16 = var16 + 1) {
                int var17 = (var13 * 2) + (var15 * 1);
                int var18 = (var14 * 1) + (var16 * 1);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (8 + var2))) || (var18 >= (10 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 10) * 8) + ((var12 * 10) * 8)) + ((var17 - var2) * 10)) + (var18 - var3)];
                }
                var9 = (((((var11 * 1) * 2) * 3) + ((var12 * 2) * 3)) + (var15 * 2)) + var16;
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



conv_runtime::ImageT<int> conv2d_dil3x2_im20x15_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  var4 = 20 + (2 * var2);
  var5 = 15 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 6) - 1) / 1) + 1;
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
              for (int var16 = 0; var16 < 2; var16 = var16 + 1) {
                int var17 = (var13 * 1) + (var15 * 3);
                int var18 = (var14 * 1) + (var16 * 2);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (20 + var2))) || (var18 >= (15 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 15) * 20) + ((var12 * 15) * 20)) + ((var17 - var2) * 15)) + (var18 - var3)];
                }
                var9 = (((((var11 * 1) * 2) * 3) + ((var12 * 2) * 3)) + (var15 * 2)) + var16;
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



conv_runtime::ImageT<int> conv2d_stride2x3_dil3x2_im20x15_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  var4 = 20 + (2 * var2);
  var5 = 15 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 6) - 1) / 2) + 1;
  var6.width = (((var5 - 2) - 1) / 3) + 1;
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
              for (int var16 = 0; var16 < 2; var16 = var16 + 1) {
                int var17 = (var13 * 2) + (var15 * 3);
                int var18 = (var14 * 3) + (var16 * 2);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (20 + var2))) || (var18 >= (15 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 15) * 20) + ((var12 * 15) * 20)) + ((var17 - var2) * 15)) + (var18 - var3)];
                }
                var9 = (((((var11 * 1) * 2) * 3) + ((var12 * 2) * 3)) + (var15 * 2)) + var16;
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



conv_runtime::ImageT<int> conv2d_pad1x2_im5x5_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 1;
  int var3 = 2;
  int var4;
  int var5;
  var4 = 5 + (2 * var2);
  var5 = 5 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 2) - 1) / 1) + 1;
  var6.width = (((var5 - 1) - 1) / 1) + 1;
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
              for (int var16 = 0; var16 < 2; var16 = var16 + 1) {
                int var17 = (var13 * 1) + (var15 * 1);
                int var18 = (var14 * 1) + (var16 * 1);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (5 + var2))) || (var18 >= (5 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 5) * 5) + ((var12 * 5) * 5)) + ((var17 - var2) * 5)) + (var18 - var3)];
                }
                var9 = (((((var11 * 1) * 2) * 3) + ((var12 * 2) * 3)) + (var15 * 2)) + var16;
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



conv_runtime::ImageT<int> conv2d_padsame_im5x5_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  assert(1);
  var4 = 7;
  var5 = 6;
  var2 = (var4 - 5) / 2;
  var3 = (var5 - 5) / 2;
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 2) - 1) / 1) + 1;
  var6.width = (((var5 - 1) - 1) / 1) + 1;
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
              for (int var16 = 0; var16 < 2; var16 = var16 + 1) {
                int var17 = (var13 * 1) + (var15 * 1);
                int var18 = (var14 * 1) + (var16 * 1);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (5 + var2))) || (var18 >= (5 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 5) * 5) + ((var12 * 5) * 5)) + ((var17 - var2) * 5)) + (var18 - var3)];
                }
                var9 = (((((var11 * 1) * 2) * 3) + ((var12 * 2) * 3)) + (var15 * 2)) + var16;
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



conv_runtime::ImageT<int> conv2d_dil3x2_stride2x3_pad3x4_im15x20_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 3;
  int var3 = 4;
  int var4;
  int var5;
  var4 = 20 + (2 * var2);
  var5 = 15 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 6) - 1) / 2) + 1;
  var6.width = (((var5 - 2) - 1) / 3) + 1;
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
              for (int var16 = 0; var16 < 2; var16 = var16 + 1) {
                int var17 = (var13 * 2) + (var15 * 3);
                int var18 = (var14 * 3) + (var16 * 2);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (20 + var2))) || (var18 >= (15 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 15) * 20) + ((var12 * 15) * 20)) + ((var17 - var2) * 15)) + (var18 - var3)];
                }
                var9 = (((((var11 * 1) * 2) * 3) + ((var12 * 2) * 3)) + (var15 * 2)) + var16;
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



conv_runtime::ImageT<int> conv2d_dil3x2_padsame_im15x20_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 0;
  int var3 = 0;
  int var4;
  int var5;
  assert(1);
  var4 = 26;
  var5 = 17;
  var2 = (var4 - 20) / 2;
  var3 = (var5 - 15) / 2;
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 6) - 1) / 1) + 1;
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
              for (int var16 = 0; var16 < 2; var16 = var16 + 1) {
                int var17 = (var13 * 1) + (var15 * 3);
                int var18 = (var14 * 1) + (var16 * 2);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (20 + var2))) || (var18 >= (15 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 15) * 20) + ((var12 * 15) * 20)) + ((var17 - var2) * 15)) + (var18 - var3)];
                }
                var9 = (((((var11 * 1) * 2) * 3) + ((var12 * 2) * 3)) + (var15 * 2)) + var16;
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



conv_runtime::ImageT<int> conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 5;
  int var3 = 4;
  int var4;
  int var5;
  var4 = 20 + (2 * var2);
  var5 = 20 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 4) - 1) / 2) + 1;
  var6.width = (((var5 - 4) - 1) / 4) + 1;
  var6.in_channels = 1;
  var6.batch_size = 5;
  var6.data = conv_runtime::conv_calloc(((var6.width * var6.height) * var6.batch_size) * var6.in_channels, 4);
  int var8;
  int var9;
  // looping over batches
  for (int var10 = 0; var10 < 5; var10 = var10 + 1) {
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
                int var17 = (var13 * 2) + (var15 * 2);
                int var18 = (var14 * 4) + (var16 * 2);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (20 + var2))) || (var18 >= (20 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 1) * 20) * 20) + ((var12 * 20) * 20)) + ((var17 - var2) * 20)) + (var18 - var3)];
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



conv_runtime::ImageT<int> conv2d_dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  int var2 = 5;
  int var3 = 4;
  int var4;
  int var5;
  var4 = 20 + (2 * var2);
  var5 = 20 + (2 * var3);
  conv_runtime::ImageT<int> var6;
  var6.height = (((var4 - 8) - 1) / 2) + 1;
  var6.width = (((var5 - 8) - 1) / 4) + 1;
  var6.in_channels = 5;
  var6.batch_size = 4;
  var6.data = conv_runtime::conv_calloc(((var6.width * var6.height) * var6.batch_size) * var6.in_channels, 4);
  int var8;
  int var9;
  // looping over batches
  for (int var10 = 0; var10 < 4; var10 = var10 + 1) {
    // looping over out channels
    for (int var11 = 0; var11 < 5; var11 = var11 + 1) {
      // looping over in channels
      for (int var12 = 0; var12 < 3; var12 = var12 + 1) {
        // looping over the output
        for (int var13 = 0; var13 < var6.height; var13 = var13 + 1) {
          for (int var14 = 0; var14 < var6.width; var14 = var14 + 1) {
            var8 = (((((var10 * 3) * var6.height) * var6.width) + ((var11 * var6.width) * var6.height)) + (var13 * var6.width)) + var14;
            // looping over the kernel
            for (int var15 = 0; var15 < 5; var15 = var15 + 1) {
              for (int var16 = 0; var16 < 5; var16 = var16 + 1) {
                int var17 = (var13 * 2) + (var15 * 2);
                int var18 = (var14 * 4) + (var16 * 2);
                int var19;
                if ((((var17 < var2) || (var18 < var3)) || (var17 >= (20 + var2))) || (var18 >= (20 + var3))) {
                  var19 = 0;
                } else {
                  var19 = arg0[(((((var10 * 3) * 20) * 20) + ((var12 * 20) * 20)) + ((var17 - var2) * 20)) + (var18 - var3)];
                }
                var9 = (((((var11 * 3) * 5) * 5) + ((var12 * 5) * 5)) + (var15 * 5)) + var16;
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



