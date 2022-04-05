#include <assert.h>

conv_runtime::ImageT<int> conv2d_default_im5x5_w3x3 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 0 - (var8 * 1);
              int var14 = 0 - (var9 * 1);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 3;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 1;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 5 + var13;
                int var21 = 3;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 5 - (var8 * 1);
                  var24 = 3;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 5 - (var8 * 1);
                  var24 = 3;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 1;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_stride2x1_im8x10_w3x2 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 0 - (var8 * 2);
              int var14 = 0 - (var9 * 1);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 3;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 1;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 8 + var13;
                int var21 = 3;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 8 - (var8 * 2);
                  var24 = 3;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 8 - (var8 * 2);
                  var24 = 3;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 1;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil3x2_im20x15_w3x2 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 0 - (var8 * 1);
              int var14 = 0 - (var9 * 1);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 9;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 3;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 20 + var13;
                int var21 = 9;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 3;
                    continue;
                  } 
                  var23 = 20 - (var8 * 1);
                  var24 = 9;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 3;
                    continue;
                  } 
                  var23 = 20 - (var8 * 1);
                  var24 = 9;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 3;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_stride2x3_dil3x2_im20x15_w3x2 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 0 - (var8 * 2);
              int var14 = 0 - (var9 * 3);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 9;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 3;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 20 + var13;
                int var21 = 9;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 3;
                    continue;
                  } 
                  var23 = 20 - (var8 * 2);
                  var24 = 9;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 3;
                    continue;
                  } 
                  var23 = 20 - (var8 * 2);
                  var24 = 9;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 3;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_pad1x2_im5x5_w3x2 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 1 - (var8 * 1);
              int var14 = 2 - (var9 * 1);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 3;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 1;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 5 + var13;
                int var21 = 3;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 7 - (var8 * 1);
                  var24 = 3;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 7 - (var8 * 1);
                  var24 = 3;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 1;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_padsame_im5x5_w3x3 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 1 - (var8 * 1);
              int var14 = 1 - (var9 * 1);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 3;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 1;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 5 + var13;
                int var21 = 3;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 7 - (var8 * 1);
                  var24 = 3;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 7 - (var8 * 1);
                  var24 = 3;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 1;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil3x2_stride2x3_pad3x4_im15x20_w3x2 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 3 - (var8 * 2);
              int var14 = 4 - (var9 * 3);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 9;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 3;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 20 + var13;
                int var21 = 9;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 3;
                    continue;
                  } 
                  var23 = 26 - (var8 * 2);
                  var24 = 9;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 3;
                    continue;
                  } 
                  var23 = 26 - (var8 * 2);
                  var24 = 9;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 3;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil3x2_padsame_im15x20_w3x3 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 3 - (var8 * 1);
              int var14 = 2 - (var9 * 1);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 9;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 3;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 20 + var13;
                int var21 = 9;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 3;
                    continue;
                  } 
                  var23 = 26 - (var8 * 1);
                  var24 = 9;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 3;
                    continue;
                  } 
                  var23 = 26 - (var8 * 1);
                  var24 = 9;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 3;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil2x2_stride2x4_pad5x4_im20x20_w3x3_batch5 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 5 - (var8 * 2);
              int var14 = 4 - (var9 * 4);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 6;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 2;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 20 + var13;
                int var21 = 6;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 2;
                    continue;
                  } 
                  var23 = 30 - (var8 * 2);
                  var24 = 6;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 2;
                    continue;
                  } 
                  var23 = 30 - (var8 * 2);
                  var24 = 6;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 2;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_dil2x2_stride2x4_pad5x4_im20x20_w5x5_batch4_inch4_outch5 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 5 - (var8 * 2);
              int var14 = 4 - (var9 * 4);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 10;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 2;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 20 + var13;
                int var21 = 10;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 2;
                    continue;
                  } 
                  var23 = 30 - (var8 * 2);
                  var24 = 10;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 2;
                    continue;
                  } 
                  var23 = 30 - (var8 * 2);
                  var24 = 10;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 2;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_im100x100_w10x10_batch10_inch10_outch10 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 0 - (var8 * 1);
              int var14 = 0 - (var9 * 1);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 10;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 1;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 100 + var13;
                int var21 = 10;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 100 - (var8 * 1);
                  var24 = 10;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 100 - (var8 * 1);
                  var24 = 10;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 1;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_stride4x4_im100x100_w10x10_batch10_inch5_outch10 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 0 - (var8 * 4);
              int var14 = 0 - (var9 * 4);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 10;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 1;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 100 + var13;
                int var21 = 10;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 100 - (var8 * 4);
                  var24 = 10;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 100 - (var8 * 4);
                  var24 = 10;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 1;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



conv_runtime::ImageT<int> conv2d_im10x10_w5x5_batch10_inch5_outch1 (int* arg0, int* arg1) {
  while (1) {
    int var23;
    int var24;
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
              int var10;
              int var11;
              int var12;
              int var13 = 0 - (var8 * 1);
              int var14 = 0 - (var9 * 1);
              var11 = 0; //Comment: looping over the kernel
              while (1) {
                int var16 = 5;
                int var17;
                if (var13 < var16) {
                  var17 = var11 < var13;
                } else {
                  var17 = var11 < var16;
                }
                if (var17) {
                  var2.data[var3] = 0;
                  var11 = var11 + 1;
                } else {
                  break;
                }
              }
              while (1) {
                int var20 = 10 + var13;
                int var21 = 5;
                if (var20 < var21) {
                  if (var11 < var20) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 10 - (var8 * 1);
                  var24 = 5;
                  int var25;
                  break;
                } else {
                  if (var11 < var21) {
                    var11 = var11 + 1;
                    continue;
                  } 
                  var23 = 10 - (var8 * 1);
                  var24 = 5;
                  break;
                }
              }
              if (var23 < var24) {
                var25 = var11 < var23;
              } else {
                var25 = var11 < var24;
              }
              if (var25) {
                var2.data[var3] = 0;
                var11 = var11 + 1;
                goto label0;
                break;
              } 
            }
          }
        }
      }
    }
    break;
  }
  return var2;
}



