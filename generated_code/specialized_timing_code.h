#include <assert.h>

conv_runtime::ImageT<int> f1 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 8;
  var2.width = 8;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(((var2.width * var2.height) * var2.batch_size) * var2.in_channels, 4);
  int var4;
  int var5;
  // looping over batches
  for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
    // looping over out channels
    for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 5; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var6 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 10) {
                for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 10) {
                    int var15 = arg0[(((((var6 * 5) * 10) * 10) + ((var8 * 10) * 10)) + ((var12 - 0) * 10)) + (var14 - 0)];
                    var5 = (((((var7 * 5) * 3) * 3) + ((var8 * 3) * 3)) + (var11 * 3)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
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
  return var2;
}



conv_runtime::ImageT<int> f2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 91;
  var2.width = 91;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(((var2.width * var2.height) * var2.batch_size) * var2.in_channels, 4);
  int var4;
  int var5;
  // looping over batches
  for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
    // looping over out channels
    for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 5; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var6 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var6 * 5) * 100) * 100) + ((var8 * 100) * 100)) + ((var12 - 0) * 100)) + (var14 - 0)];
                    var5 = (((((var7 * 5) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
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
  return var2;
}



conv_runtime::ImageT<int> f3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 23;
  var2.width = 23;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(((var2.width * var2.height) * var2.batch_size) * var2.in_channels, 4);
  int var4;
  int var5;
  // looping over batches
  for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
    // looping over out channels
    for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 5; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var6 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var6 * 5) * 100) * 100) + ((var8 * 100) * 100)) + ((var12 - 0) * 100)) + (var14 - 0)];
                    var5 = (((((var7 * 5) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
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
  return var2;
}



conv_runtime::ImageT<int> f4 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 91;
  var2.width = 91;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(((var2.width * var2.height) * var2.batch_size) * var2.in_channels, 4);
  int var4;
  int var5;
  // looping over batches
  for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
    // looping over out channels
    for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 10; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var6 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var6 * 10) * 100) * 100) + ((var8 * 100) * 100)) + ((var12 - 0) * 100)) + (var14 - 0)];
                    var5 = (((((var7 * 10) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
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
  return var2;
}



conv_runtime::ImageT<int> f5 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 91;
  var2.width = 91;
  var2.in_channels = 1;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(((var2.width * var2.height) * var2.batch_size) * var2.in_channels, 4);
  int var4;
  int var5;
  // looping over batches
  for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
    // looping over out channels
    for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 10; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var6 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var6 * 10) * 100) * 100) + ((var8 * 100) * 100)) + ((var12 - 0) * 100)) + (var14 - 0)];
                    var5 = (((((var7 * 10) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
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
  return var2;
}



conv_runtime::ImageT<int> f6 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 991;
  var2.width = 991;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(((var2.width * var2.height) * var2.batch_size) * var2.in_channels, 4);
  int var4;
  int var5;
  // looping over batches
  for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
    // looping over out channels
    for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 10; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var6 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 1000) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 1000) {
                    int var15 = arg0[(((((var6 * 10) * 1000) * 1000) + ((var8 * 1000) * 1000)) + ((var12 - 0) * 1000)) + (var14 - 0)];
                    var5 = (((((var7 * 10) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
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
  return var2;
}



conv_runtime::ImageT<int> f7 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 991;
  var2.width = 991;
  var2.in_channels = 100;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(((var2.width * var2.height) * var2.batch_size) * var2.in_channels, 4);
  int var4;
  int var5;
  // looping over batches
  for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
    // looping over out channels
    for (int var7 = 0; var7 < 100; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 10; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var6 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 1000) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 1000) {
                    int var15 = arg0[(((((var6 * 10) * 1000) * 1000) + ((var8 * 1000) * 1000)) + ((var12 - 0) * 1000)) + (var14 - 0)];
                    var5 = (((((var7 * 10) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
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
  return var2;
}



conv_runtime::ImageT<int> f8 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 991;
  var2.width = 991;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(((var2.width * var2.height) * var2.batch_size) * var2.in_channels, 4);
  int var4;
  int var5;
  // looping over batches
  for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
    // looping over out channels
    for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
      // looping over in channels
      for (int var8 = 0; var8 < 100; var8 = var8 + 1) {
        // looping over the output
        for (int var9 = 0; var9 < var2.height; var9 = var9 + 1) {
          for (int var10 = 0; var10 < var2.width; var10 = var10 + 1) {
            var4 = (((((var6 * var2.in_channels) * var2.height) * var2.width) + ((var7 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            // looping over the kernel
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              if (var12 < 0) {
                continue;
              } 
              if (var12 < 1000) {
                for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 1);
                  if (var14 < 0) {
                    continue;
                  } 
                  if (var14 < 1000) {
                    int var15 = arg0[(((((var6 * 100) * 1000) * 1000) + ((var8 * 1000) * 1000)) + ((var12 - 0) * 1000)) + (var14 - 0)];
                    var5 = (((((var7 * 100) * 10) * 10) + ((var8 * 10) * 10)) + (var11 * 10)) + var13;
                    var2.data[var4] = var2.data[var4] + (var15 * arg1[var5]);
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
  return var2;
}



