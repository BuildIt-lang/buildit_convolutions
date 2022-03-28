#include <assert.h>

conv_runtime::ImageT<int> conv2d_default_im5x5_w3x3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 3;
  var2.width = 3;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(9, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 5) {
                for (int var12 = 0; var12 < 3; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 5) {
                    int var14 = arg0[(((((var5 * 1) * 5) * 5) + ((var7 * 5) * 5)) + ((var11 - 0) * 5)) + (var13 - 0)];
                    var4 = (((((var6 * 1) * 3) * 3) + ((var7 * 3) * 3)) + (var10 * 3)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_stride2x1_im8x10_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 3;
  var2.width = 9;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(27, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 2) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 8) {
                for (int var12 = 0; var12 < 2; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 10) {
                    int var14 = arg0[(((((var5 * 1) * 10) * 8) + ((var7 * 10) * 8)) + ((var11 - 0) * 10)) + (var13 - 0)];
                    var4 = (((((var6 * 1) * 2) * 3) + ((var7 * 2) * 3)) + (var10 * 2)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_dil3x2_im20x15_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 14;
  var2.width = 13;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(182, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 3);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 20) {
                for (int var12 = 0; var12 < 2; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 2);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 15) {
                    int var14 = arg0[(((((var5 * 1) * 15) * 20) + ((var7 * 15) * 20)) + ((var11 - 0) * 15)) + (var13 - 0)];
                    var4 = (((((var6 * 1) * 2) * 3) + ((var7 * 2) * 3)) + (var10 * 2)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_stride2x3_dil3x2_im20x15_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 7;
  var2.width = 5;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(35, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 2) + (var10 * 3);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 20) {
                for (int var12 = 0; var12 < 2; var12 = var12 + 1) {
                  int var13 = (var9 * 3) + (var12 * 2);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 15) {
                    int var14 = arg0[(((((var5 * 1) * 15) * 20) + ((var7 * 15) * 20)) + ((var11 - 0) * 15)) + (var13 - 0)];
                    var4 = (((((var6 * 1) * 2) * 3) + ((var7 * 2) * 3)) + (var10 * 2)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_pad1x2_im5x5_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 5;
  var2.width = 8;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(40, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 1) {
                continue;
              } 
              if (var11 < 6) {
                for (int var12 = 0; var12 < 2; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 2) {
                    continue;
                  } 
                  if (var13 < 7) {
                    int var14 = arg0[(((((var5 * 1) * 5) * 5) + ((var7 * 5) * 5)) + ((var11 - 1) * 5)) + (var13 - 2)];
                    var4 = (((((var6 * 1) * 2) * 3) + ((var7 * 2) * 3)) + (var10 * 2)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 1) {
                continue;
              } 
              if (var11 < 6) {
                for (int var12 = 0; var12 < 3; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 1) {
                    continue;
                  } 
                  if (var13 < 6) {
                    int var14 = arg0[(((((var5 * 1) * 5) * 5) + ((var7 * 5) * 5)) + ((var11 - 1) * 5)) + (var13 - 1)];
                    var4 = (((((var6 * 1) * 3) * 3) + ((var7 * 3) * 3)) + (var10 * 3)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_dil3x2_stride2x3_pad3x4_im15x20_w3x2 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 10;
  var2.width = 7;
  var2.in_channels = 1;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(70, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 2) + (var10 * 3);
              if (var11 < 3) {
                continue;
              } 
              if (var11 < 23) {
                for (int var12 = 0; var12 < 2; var12 = var12 + 1) {
                  int var13 = (var9 * 3) + (var12 * 2);
                  if (var13 < 4) {
                    continue;
                  } 
                  if (var13 < 19) {
                    int var14 = arg0[(((((var5 * 1) * 15) * 20) + ((var7 * 15) * 20)) + ((var11 - 3) * 15)) + (var13 - 4)];
                    var4 = (((((var6 * 1) * 2) * 3) + ((var7 * 2) * 3)) + (var10 * 2)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 3);
              if (var11 < 3) {
                continue;
              } 
              if (var11 < 23) {
                for (int var12 = 0; var12 < 3; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 2);
                  if (var13 < 2) {
                    continue;
                  } 
                  if (var13 < 17) {
                    int var14 = arg0[(((((var5 * 1) * 15) * 20) + ((var7 * 15) * 20)) + ((var11 - 3) * 15)) + (var13 - 2)];
                    var4 = (((((var6 * 1) * 3) * 3) + ((var7 * 3) * 3)) + (var10 * 3)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 13;
  var2.width = 6;
  var2.in_channels = 1;
  var2.batch_size = 5;
  var2.data = conv_runtime::conv_calloc(390, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 5; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 1; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
              int var11 = (var8 * 2) + (var10 * 2);
              if (var11 < 5) {
                continue;
              } 
              if (var11 < 25) {
                for (int var12 = 0; var12 < 3; var12 = var12 + 1) {
                  int var13 = (var9 * 4) + (var12 * 2);
                  if (var13 < 4) {
                    continue;
                  } 
                  if (var13 < 24) {
                    int var14 = arg0[(((((var5 * 1) * 20) * 20) + ((var7 * 20) * 20)) + ((var11 - 5) * 20)) + (var13 - 4)];
                    var4 = (((((var6 * 1) * 3) * 3) + ((var7 * 3) * 3)) + (var10 * 3)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 11;
  var2.width = 5;
  var2.in_channels = 5;
  var2.batch_size = 4;
  var2.data = conv_runtime::conv_calloc(1100, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 4; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 5; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 3; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 5; var10 = var10 + 1) {
              int var11 = (var8 * 2) + (var10 * 2);
              if (var11 < 5) {
                continue;
              } 
              if (var11 < 25) {
                for (int var12 = 0; var12 < 5; var12 = var12 + 1) {
                  int var13 = (var9 * 4) + (var12 * 2);
                  if (var13 < 4) {
                    continue;
                  } 
                  if (var13 < 24) {
                    int var14 = arg0[(((((var5 * 3) * 20) * 20) + ((var7 * 20) * 20)) + ((var11 - 5) * 20)) + (var13 - 4)];
                    var4 = (((((var6 * 3) * 5) * 5) + ((var7 * 5) * 5)) + (var10 * 5)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_im100x100_w10x10_batch10_inch10_outch10 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 91;
  var2.width = 91;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(828100, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 10; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
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
                    int var14 = arg0[(((((var5 * 10) * 100) * 100) + ((var7 * 100) * 100)) + ((var11 - 0) * 100)) + (var13 - 0)];
                    var4 = (((((var6 * 10) * 10) * 10) + ((var7 * 10) * 10)) + (var10 * 10)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_stride4x4_im100x100_w10x10_batch10_inch5_outch10 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 23;
  var2.width = 23;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(52900, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 10; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 5; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 10; var10 = var10 + 1) {
              int var11 = (var8 * 4) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 100) {
                for (int var12 = 0; var12 < 10; var12 = var12 + 1) {
                  int var13 = (var9 * 4) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 100) {
                    int var14 = arg0[(((((var5 * 5) * 100) * 100) + ((var7 * 100) * 100)) + ((var11 - 0) * 100)) + (var13 - 0)];
                    var4 = (((((var6 * 5) * 10) * 10) + ((var7 * 10) * 10)) + (var10 * 10)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



conv_runtime::ImageT<int> conv2d_im10x10_w5x5_batch10_inch5_outch1 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 6;
  var2.width = 6;
  var2.in_channels = 1;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(360, 4);
  int var3;
  int var4;
  // looping over batches
  for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
    // looping over out channels
    for (int var6 = 0; var6 < 1; var6 = var6 + 1) {
      // looping over in channels
      for (int var7 = 0; var7 < 5; var7 = var7 + 1) {
        // looping over the output
        for (int var8 = 0; var8 < var2.height; var8 = var8 + 1) {
          for (int var9 = 0; var9 < var2.width; var9 = var9 + 1) {
            var3 = (((((var5 * var2.in_channels) * var2.height) * var2.width) + ((var6 * var2.width) * var2.height)) + (var8 * var2.width)) + var9;
            // looping over the kernel
            for (int var10 = 0; var10 < 5; var10 = var10 + 1) {
              int var11 = (var8 * 1) + (var10 * 1);
              if (var11 < 0) {
                continue;
              } 
              if (var11 < 10) {
                for (int var12 = 0; var12 < 5; var12 = var12 + 1) {
                  int var13 = (var9 * 1) + (var12 * 1);
                  if (var13 < 0) {
                    continue;
                  } 
                  if (var13 < 10) {
                    int var14 = arg0[(((((var5 * 5) * 10) * 10) + ((var7 * 10) * 10)) + ((var11 - 0) * 10)) + (var13 - 0)];
                    var4 = (((((var6 * 5) * 5) * 5) + ((var7 * 5) * 5)) + (var10 * 5)) + var12;
                    var2.data[var3] = var2.data[var3] + (var14 * arg1[var4]);
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



