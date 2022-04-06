#include <assert.h>

conv_runtime::ImageT<int> f1 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 91;
  var2.width = 91;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(828100, 4);
  #pragma omp parallel for
  //  looping over batches
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    #pragma omp parallel for
    //  looping over out channels
    for (int var4 = 0; var4 < 10; var4 = var4 + 1) {
      #pragma omp parallel for
      //  looping over in channels
      for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 100) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 100) {
                    int var15 = arg0[(((((var3 * 10) * 100) * 100) + ((var5 * 100) * 100)) + ((var12 - 0) * 100)) + (var14 - 0)];
                    var7 = (((((var4 * 10) * 10) * 10) + ((var5 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                    var8 = var8 + 1;
                  } else {
                    break;
                  }
                }
              } else {
                break;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 10) * 10) * 10;
  return var2;
}



conv_runtime::ImageT<int> f2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 91;
  var2.width = 91;
  var2.in_channels = 10;
  var2.batch_size = 20;
  var2.data = conv_runtime::conv_calloc(1656200, 4);
  #pragma omp parallel for
  //  looping over batches
  for (int var3 = 0; var3 < 20; var3 = var3 + 1) {
    #pragma omp parallel for
    //  looping over out channels
    for (int var4 = 0; var4 < 10; var4 = var4 + 1) {
      #pragma omp parallel for
      //  looping over in channels
      for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 100) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 100) {
                    int var15 = arg0[(((((var3 * 10) * 100) * 100) + ((var5 * 100) * 100)) + ((var12 - 0) * 100)) + (var14 - 0)];
                    var7 = (((((var4 * 10) * 10) * 10) + ((var5 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                    var8 = var8 + 1;
                  } else {
                    break;
                  }
                }
              } else {
                break;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 20) * 10) * 10;
  return var2;
}



conv_runtime::ImageT<int> f3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 191;
  var2.width = 191;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(3648100, 4);
  #pragma omp parallel for
  //  looping over batches
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    #pragma omp parallel for
    //  looping over out channels
    for (int var4 = 0; var4 < 10; var4 = var4 + 1) {
      #pragma omp parallel for
      //  looping over in channels
      for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 200) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 200) {
                    int var15 = arg0[(((((var3 * 10) * 200) * 200) + ((var5 * 200) * 200)) + ((var12 - 0) * 200)) + (var14 - 0)];
                    var7 = (((((var4 * 10) * 10) * 10) + ((var5 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                    var8 = var8 + 1;
                  } else {
                    break;
                  }
                }
              } else {
                break;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 10) * 10) * 10;
  return var2;
}



conv_runtime::ImageT<int> f4 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 191;
  var2.width = 191;
  var2.in_channels = 10;
  var2.batch_size = 20;
  var2.data = conv_runtime::conv_calloc(7296200, 4);
  #pragma omp parallel for
  //  looping over batches
  for (int var3 = 0; var3 < 20; var3 = var3 + 1) {
    #pragma omp parallel for
    //  looping over out channels
    for (int var4 = 0; var4 < 10; var4 = var4 + 1) {
      #pragma omp parallel for
      //  looping over in channels
      for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 200) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 200) {
                    int var15 = arg0[(((((var3 * 10) * 200) * 200) + ((var5 * 200) * 200)) + ((var12 - 0) * 200)) + (var14 - 0)];
                    var7 = (((((var4 * 10) * 10) * 10) + ((var5 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                    var8 = var8 + 1;
                  } else {
                    break;
                  }
                }
              } else {
                break;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 20) * 10) * 10;
  return var2;
}



conv_runtime::ImageT<int> f5 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 291;
  var2.width = 291;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(8468100, 4);
  #pragma omp parallel for
  //  looping over batches
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    #pragma omp parallel for
    //  looping over out channels
    for (int var4 = 0; var4 < 10; var4 = var4 + 1) {
      #pragma omp parallel for
      //  looping over in channels
      for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 300) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 300) {
                    int var15 = arg0[(((((var3 * 10) * 300) * 300) + ((var5 * 300) * 300)) + ((var12 - 0) * 300)) + (var14 - 0)];
                    var7 = (((((var4 * 10) * 10) * 10) + ((var5 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                    var8 = var8 + 1;
                  } else {
                    break;
                  }
                }
              } else {
                break;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 10) * 10) * 10;
  return var2;
}



conv_runtime::ImageT<int> f6 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 291;
  var2.width = 291;
  var2.in_channels = 10;
  var2.batch_size = 20;
  var2.data = conv_runtime::conv_calloc(16936200, 4);
  #pragma omp parallel for
  //  looping over batches
  for (int var3 = 0; var3 < 20; var3 = var3 + 1) {
    #pragma omp parallel for
    //  looping over out channels
    for (int var4 = 0; var4 < 10; var4 = var4 + 1) {
      #pragma omp parallel for
      //  looping over in channels
      for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 300) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 300) {
                    int var15 = arg0[(((((var3 * 10) * 300) * 300) + ((var5 * 300) * 300)) + ((var12 - 0) * 300)) + (var14 - 0)];
                    var7 = (((((var4 * 10) * 10) * 10) + ((var5 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                    var8 = var8 + 1;
                  } else {
                    break;
                  }
                }
              } else {
                break;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 20) * 10) * 10;
  return var2;
}



