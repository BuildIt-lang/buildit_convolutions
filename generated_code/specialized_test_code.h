#include <assert.h>

#include <omp.h>

conv_runtime::ImageT<int> conv2d_default_im5x5_w3x3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 3;
  var2.width = 3;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(9, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 5) {
                for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 5) {
                    int var15 = arg0[(((((var3 * 1) * 5) * 5) + ((var8 * 5) * 5)) + ((var12 - 0) * 5)) + (var14 - 0)];
                    var5 = (((((var7 * 1) * 3) * 3) + ((var8 * 3) * 3)) + (var11 * 3)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_stride2x1_im8x10_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 3;
  var2.width = 9;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(27, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 2) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 8) {
                for (int var13 = 0; var13 < 2; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 10) {
                    int var15 = arg0[(((((var3 * 1) * 10) * 8) + ((var8 * 10) * 8)) + ((var12 - 0) * 10)) + (var14 - 0)];
                    var5 = (((((var7 * 1) * 2) * 3) + ((var8 * 2) * 3)) + (var11 * 2)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil3x2_im20x15_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 14;
  var2.width = 13;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(182, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 3);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 20) {
                for (int var13 = 0; var13 < 2; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 2);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 15) {
                    int var15 = arg0[(((((var3 * 1) * 15) * 20) + ((var8 * 15) * 20)) + ((var12 - 0) * 15)) + (var14 - 0)];
                    var5 = (((((var7 * 1) * 2) * 3) + ((var8 * 2) * 3)) + (var11 * 2)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_stride2x3_dil3x2_im20x15_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 7;
  var2.width = 5;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(35, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 2) + (var11 * 3);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 20) {
                for (int var13 = 0; var13 < 2; var13 = var13 + 1) {
                  int var14 = (var10 * 3) + (var13 * 2);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 15) {
                    int var15 = arg0[(((((var3 * 1) * 15) * 20) + ((var8 * 15) * 20)) + ((var12 - 0) * 15)) + (var14 - 0)];
                    var5 = (((((var7 * 1) * 2) * 3) + ((var8 * 2) * 3)) + (var11 * 2)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_pad1x2_im5x5_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 5;
  var2.width = 8;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(40, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 1) {
                continue;
              } 
              if (var12 < 6) {
                for (int var13 = 0; var13 < 2; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 2) {
                    continue;
                  } 
                  if (var14 < 7) {
                    int var15 = arg0[(((((var3 * 1) * 5) * 5) + ((var8 * 5) * 5)) + ((var12 - 1) * 5)) + (var14 - 2)];
                    var5 = (((((var7 * 1) * 2) * 3) + ((var8 * 2) * 3)) + (var11 * 2)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_padsame_im5x5_w3x3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 5;
  var2.width = 5;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(25, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 1) {
                continue;
              } 
              if (var12 < 6) {
                for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 1) {
                    continue;
                  } 
                  if (var14 < 6) {
                    int var15 = arg0[(((((var3 * 1) * 5) * 5) + ((var8 * 5) * 5)) + ((var12 - 1) * 5)) + (var14 - 1)];
                    var5 = (((((var7 * 1) * 3) * 3) + ((var8 * 3) * 3)) + (var11 * 3)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil3x2_stride2x3_pad3x4_im15x20_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 10;
  var2.width = 7;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(70, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 2) + (var11 * 3);
              if (var12 < 3) {
                continue;
              } 
              if (var12 < 23) {
                for (int var13 = 0; var13 < 2; var13 = var13 + 1) {
                  int var14 = (var10 * 3) + (var13 * 2);
                  if (var14 < 4) {
                    continue;
                  } 
                  if (var14 < 19) {
                    int var15 = arg0[(((((var3 * 1) * 15) * 20) + ((var8 * 15) * 20)) + ((var12 - 3) * 15)) + (var14 - 4)];
                    var5 = (((((var7 * 1) * 2) * 3) + ((var8 * 2) * 3)) + (var11 * 2)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil3x2_padsame_im15x20_w3x3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 20;
  var2.width = 15;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(300, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 3);
              if (var12 < 3) {
                continue;
              } 
              if (var12 < 23) {
                for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 2);
                  if (var14 < 2) {
                    continue;
                  } 
                  if (var14 < 17) {
                    int var15 = arg0[(((((var3 * 1) * 15) * 20) + ((var8 * 15) * 20)) + ((var12 - 3) * 15)) + (var14 - 2)];
                    var5 = (((((var7 * 1) * 3) * 3) + ((var8 * 3) * 3)) + (var11 * 3)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 13;
  var2.width = 6;
  var2.in_channels = 1;
  var2.batch_size = 5;
  var2.data = conv_runtime::conv_calloc(390, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 5; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 1; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 2) + (var11 * 2);
              if (var12 < 5) {
                continue;
              } 
              if (var12 < 25) {
                for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
                  int var14 = (var10 * 4) + (var13 * 2);
                  if (var14 < 4) {
                    continue;
                  } 
                  if (var14 < 24) {
                    int var15 = arg0[(((((var3 * 1) * 20) * 20) + ((var8 * 20) * 20)) + ((var12 - 5) * 20)) + (var14 - 4)];
                    var5 = (((((var7 * 1) * 3) * 3) + ((var8 * 3) * 3)) + (var11 * 3)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 5;
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 11;
  var2.width = 5;
  var2.in_channels = 5;
  var2.batch_size = 4;
  var2.data = conv_runtime::conv_calloc(1100, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 4; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 5; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 3; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 5; var11 = var11 + 1) {
              int var12 = (var9 * 2) + (var11 * 2);
              if (var12 < 5) {
                continue;
              } 
              if (var12 < 25) {
                for (int var13 = 0; var13 < 5; var13 = var13 + 1) {
                  int var14 = (var10 * 4) + (var13 * 2);
                  if (var14 < 4) {
                    continue;
                  } 
                  if (var14 < 24) {
                    int var15 = arg0[(((((var3 * 3) * 20) * 20) + ((var8 * 20) * 20)) + ((var12 - 5) * 20)) + (var14 - 4)];
                    var5 = (((((var7 * 3) * 5) * 5) + ((var8 * 5) * 5)) + (var11 * 5)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 4;
  return var2;
}



conv_runtime::ImageT<int> conv2d_im100x100_w10x10_batch10_inch10_outch10 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 91;
  var2.width = 91;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(828100, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 10; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var3 * 10) * 100) * 100) + ((var8 * 100) * 100)) + ((var12 - 0) * 100)) + (var14 - 0)];
                    var5 = (((((var7 * 10) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 10;
  return var2;
}



conv_runtime::ImageT<int> conv2d_stride4x4_im100x100_w10x10_batch10_inch5_outch10 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 23;
  var2.width = 23;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(52900, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 5; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 4) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 100) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 4) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 100) {
                    int var15 = arg0[(((((var3 * 5) * 100) * 100) + ((var8 * 100) * 100)) + ((var12 - 0) * 100)) + (var14 - 0)];
                    var5 = (((((var7 * 5) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 10;
  return var2;
}



conv_runtime::ImageT<int> conv2d_im10x10_w5x5_batch10_inch5_outch1 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 6;
  var2.width = 6;
  var2.in_channels = 1;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(360, 4);
  #pragma omp parallel
  {
  #pragma omp for
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    int var4;
    int var5;
    int var6 = 0;
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 5; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 5; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 10) {
                for (int var13 = 0; var13 < 5; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 10) {
                    int var15 = arg0[(((((var3 * 5) * 10) * 10) + ((var8 * 10) * 10)) + ((var12 - 0) * 10)) + (var14 - 0)];
                    var5 = (((((var7 * 5) * 5) * 5) + ((var8 * 5) * 5)) + (var11 * 5)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
                    var6 = var6 + 1;
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
    var2.mult_cnt = var6;
  }
  }
  
  var2.mult_cnt = var2.mult_cnt * 10;
  return var2;
}



