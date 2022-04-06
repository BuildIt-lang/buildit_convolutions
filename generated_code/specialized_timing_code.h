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
  var2.mult_cnt = 0;
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    int var4;
    int var5;
    // looping over out channels
    for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 100) {
                for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 100) {
                    int var14 = arg0[(((((var3 * 10) * 100) * 100) + ((var7 * 100) * 100)) + ((var11 - 0) * 100)) + (var13 - 0)];
                    var5 = (((((var6 * 10) * 10) * 10) + ((var7 * 10) * 10)) + (var10 * 10)) + var12;
                    var2.data[var4] = var2.data[var4] + (var14 * arg1[var5]);
                    var2.mult_cnt = var2.mult_cnt + 1;
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
      }
    }
  }
  }
  
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
  var2.mult_cnt = 0;
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 20; var3 = var3 + 1) {
    int var4;
    int var5;
    // looping over out channels
    for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 100) {
                for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 100) {
                    int var14 = arg0[(((((var3 * 10) * 100) * 100) + ((var7 * 100) * 100)) + ((var11 - 0) * 100)) + (var13 - 0)];
                    var5 = (((((var6 * 10) * 10) * 10) + ((var7 * 10) * 10)) + (var10 * 10)) + var12;
                    var2.data[var4] = var2.data[var4] + (var14 * arg1[var5]);
                    var2.mult_cnt = var2.mult_cnt + 1;
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
      }
    }
  }
  }
  
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
  var2.mult_cnt = 0;
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    int var4;
    int var5;
    // looping over out channels
    for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 200) {
                for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 200) {
                    int var14 = arg0[(((((var3 * 10) * 200) * 200) + ((var7 * 200) * 200)) + ((var11 - 0) * 200)) + (var13 - 0)];
                    var5 = (((((var6 * 10) * 10) * 10) + ((var7 * 10) * 10)) + (var10 * 10)) + var12;
                    var2.data[var4] = var2.data[var4] + (var14 * arg1[var5]);
                    var2.mult_cnt = var2.mult_cnt + 1;
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
      }
    }
  }
  }
  
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
  var2.mult_cnt = 0;
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 20; var3 = var3 + 1) {
    int var4;
    int var5;
    // looping over out channels
    for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 200) {
                for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 200) {
                    int var14 = arg0[(((((var3 * 10) * 200) * 200) + ((var7 * 200) * 200)) + ((var11 - 0) * 200)) + (var13 - 0)];
                    var5 = (((((var6 * 10) * 10) * 10) + ((var7 * 10) * 10)) + (var10 * 10)) + var12;
                    var2.data[var4] = var2.data[var4] + (var14 * arg1[var5]);
                    var2.mult_cnt = var2.mult_cnt + 1;
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
      }
    }
  }
  }
  
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
  var2.mult_cnt = 0;
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    int var4;
    int var5;
    // looping over out channels
    for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 300) {
                for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 300) {
                    int var14 = arg0[(((((var3 * 10) * 300) * 300) + ((var7 * 300) * 300)) + ((var11 - 0) * 300)) + (var13 - 0)];
                    var5 = (((((var6 * 10) * 10) * 10) + ((var7 * 10) * 10)) + (var10 * 10)) + var12;
                    var2.data[var4] = var2.data[var4] + (var14 * arg1[var5]);
                    var2.mult_cnt = var2.mult_cnt + 1;
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
      }
    }
  }
  }
  
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
  var2.mult_cnt = 0;
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 20; var3 = var3 + 1) {
    int var4;
    int var5;
    // looping over out channels
    for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 300) {
                for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 300) {
                    int var14 = arg0[(((((var3 * 10) * 300) * 300) + ((var7 * 300) * 300)) + ((var11 - 0) * 300)) + (var13 - 0)];
                    var5 = (((((var6 * 10) * 10) * 10) + ((var7 * 10) * 10)) + (var10 * 10)) + var12;
                    var2.data[var4] = var2.data[var4] + (var14 * arg1[var5]);
                    var2.mult_cnt = var2.mult_cnt + 1;
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
      }
    }
  }
  }
  
  return var2;
}



