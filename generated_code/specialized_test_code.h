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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 3; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 3; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
                int var15 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var12 - 0) * 5)) + (((var10 * 1) + (var13 * 1)) - 0)];
                var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var11 * 3)) + var13;
                var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 1) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 3; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 9; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 2) + (var11 * 1);
              for (int var13 = 0; var13 < 2; var13 = var13 + 1) {
                int var15 = arg0[(((((var3 * 1) * 10) * 8) + ((var5 * 10) * 8)) + ((var12 - 0) * 10)) + (((var10 * 1) + (var13 * 1)) - 0)];
                var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var11 * 2)) + var13;
                var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 1) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 14; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 13; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 3);
              for (int var13 = 0; var13 < 2; var13 = var13 + 1) {
                int var15 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var12 - 0) * 15)) + (((var10 * 1) + (var13 * 2)) - 0)];
                var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var11 * 2)) + var13;
                var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 1) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 7; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 5; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 2) + (var11 * 3);
              for (int var13 = 0; var13 < 2; var13 = var13 + 1) {
                int var15 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var12 - 0) * 15)) + (((var10 * 3) + (var13 * 2)) - 0)];
                var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var11 * 2)) + var13;
                var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 1) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 1; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 1; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var12 - 1) * 5)) + (var14 - 2)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var11 * 2)) + var13;
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
          for (int var16 = 1; var16 < 2; var16 = var16 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var16;
            for (int var17 = 0; var17 < 3; var17 = var17 + 1) {
              int var18 = (var9 * 1) + (var17 * 1);
              if (var18 < 1) {
                continue;
              } 
              if (var18 < 6) {
                for (int var19 = 0; var19 < 2; var19 = var19 + 1) {
                  int var20 = (var16 * 1) + (var19 * 1);
                  if (var20 < 2) {
                    continue;
                  } 
                  if (var20 < 7) {
                    int var21 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var18 - 1) * 5)) + (var20 - 2)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var17 * 2)) + var19;
                    var2.data[var6] = var2.data[var6] + (var21 * arg1[var7]);
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
          for (int var22 = 2; var22 < 6; var22 = var22 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var22;
            for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
              int var24 = (var9 * 1) + (var23 * 1);
              if (var24 < 1) {
                continue;
              } 
              if (var24 < 6) {
                for (int var25 = 0; var25 < 2; var25 = var25 + 1) {
                  int var27 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var24 - 1) * 5)) + (((var22 * 1) + (var25 * 1)) - 2)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var23 * 2)) + var25;
                  var2.data[var6] = var2.data[var6] + (var27 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var28 = 6; var28 < 8; var28 = var28 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var28;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = (var9 * 1) + (var29 * 1);
              if (var30 < 1) {
                continue;
              } 
              if (var30 < 6) {
                for (int var31 = 0; var31 < 2; var31 = var31 + 1) {
                  int var32 = (var28 * 1) + (var31 * 1);
                  if (var32 < 2) {
                    continue;
                  } 
                  if (var32 < 7) {
                    int var33 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var30 - 1) * 5)) + (var32 - 2)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var29 * 2)) + var31;
                    var2.data[var6] = var2.data[var6] + (var33 * arg1[var7]);
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
        // looping over the output
        for (int var34 = 1; var34 < 4; var34 = var34 + 1) {
          for (int var35 = 0; var35 < 1; var35 = var35 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var35;
            for (int var36 = 0; var36 < 3; var36 = var36 + 1) {
              int var37 = (var34 * 1) + (var36 * 1);
              for (int var38 = 0; var38 < 2; var38 = var38 + 1) {
                int var39 = (var35 * 1) + (var38 * 1);
                if (var39 < 2) {
                  continue;
                } 
                if (var39 < 7) {
                  int var40 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var37 - 1) * 5)) + (var39 - 2)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var36 * 2)) + var38;
                  var2.data[var6] = var2.data[var6] + (var40 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var41 = 1; var41 < 2; var41 = var41 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var41;
            for (int var42 = 0; var42 < 3; var42 = var42 + 1) {
              int var43 = (var34 * 1) + (var42 * 1);
              for (int var44 = 0; var44 < 2; var44 = var44 + 1) {
                int var45 = (var41 * 1) + (var44 * 1);
                if (var45 < 2) {
                  continue;
                } 
                if (var45 < 7) {
                  int var46 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var43 - 1) * 5)) + (var45 - 2)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var42 * 2)) + var44;
                  var2.data[var6] = var2.data[var6] + (var46 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var47 = 2; var47 < 6; var47 = var47 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var47;
            for (int var48 = 0; var48 < 3; var48 = var48 + 1) {
              int var49 = (var34 * 1) + (var48 * 1);
              for (int var50 = 0; var50 < 2; var50 = var50 + 1) {
                int var52 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var49 - 1) * 5)) + (((var47 * 1) + (var50 * 1)) - 2)];
                var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var48 * 2)) + var50;
                var2.data[var6] = var2.data[var6] + (var52 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var53 = 6; var53 < 8; var53 = var53 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var53;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = (var34 * 1) + (var54 * 1);
              for (int var56 = 0; var56 < 2; var56 = var56 + 1) {
                int var57 = (var53 * 1) + (var56 * 1);
                if (var57 < 2) {
                  continue;
                } 
                if (var57 < 7) {
                  int var58 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var55 - 1) * 5)) + (var57 - 2)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var54 * 2)) + var56;
                  var2.data[var6] = var2.data[var6] + (var58 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var59 = 4; var59 < 5; var59 = var59 + 1) {
          for (int var60 = 0; var60 < 1; var60 = var60 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var60;
            for (int var61 = 0; var61 < 3; var61 = var61 + 1) {
              int var62 = (var59 * 1) + (var61 * 1);
              if (var62 < 1) {
                continue;
              } 
              if (var62 < 6) {
                for (int var63 = 0; var63 < 2; var63 = var63 + 1) {
                  int var64 = (var60 * 1) + (var63 * 1);
                  if (var64 < 2) {
                    continue;
                  } 
                  if (var64 < 7) {
                    int var65 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var62 - 1) * 5)) + (var64 - 2)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var61 * 2)) + var63;
                    var2.data[var6] = var2.data[var6] + (var65 * arg1[var7]);
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
          for (int var66 = 1; var66 < 2; var66 = var66 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var66;
            for (int var67 = 0; var67 < 3; var67 = var67 + 1) {
              int var68 = (var59 * 1) + (var67 * 1);
              if (var68 < 1) {
                continue;
              } 
              if (var68 < 6) {
                for (int var69 = 0; var69 < 2; var69 = var69 + 1) {
                  int var70 = (var66 * 1) + (var69 * 1);
                  if (var70 < 2) {
                    continue;
                  } 
                  if (var70 < 7) {
                    int var71 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var68 - 1) * 5)) + (var70 - 2)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var67 * 2)) + var69;
                    var2.data[var6] = var2.data[var6] + (var71 * arg1[var7]);
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
          for (int var72 = 2; var72 < 6; var72 = var72 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var72;
            for (int var73 = 0; var73 < 3; var73 = var73 + 1) {
              int var74 = (var59 * 1) + (var73 * 1);
              if (var74 < 1) {
                continue;
              } 
              if (var74 < 6) {
                for (int var75 = 0; var75 < 2; var75 = var75 + 1) {
                  int var77 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var74 - 1) * 5)) + (((var72 * 1) + (var75 * 1)) - 2)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var73 * 2)) + var75;
                  var2.data[var6] = var2.data[var6] + (var77 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var78 = 6; var78 < 8; var78 = var78 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var78;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = (var59 * 1) + (var79 * 1);
              if (var80 < 1) {
                continue;
              } 
              if (var80 < 6) {
                for (int var81 = 0; var81 < 2; var81 = var81 + 1) {
                  int var82 = (var78 * 1) + (var81 * 1);
                  if (var82 < 2) {
                    continue;
                  } 
                  if (var82 < 7) {
                    int var83 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var80 - 1) * 5)) + (var82 - 2)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var79 * 2)) + var81;
                    var2.data[var6] = var2.data[var6] + (var83 * arg1[var7]);
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
  var2.mult_cnt = ((var2.mult_cnt * 1) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 1; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 1; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var12 - 1) * 5)) + (var14 - 1)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var11 * 3)) + var13;
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
          for (int var16 = 1; var16 < 4; var16 = var16 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var16;
            for (int var17 = 0; var17 < 3; var17 = var17 + 1) {
              int var18 = (var9 * 1) + (var17 * 1);
              if (var18 < 1) {
                continue;
              } 
              if (var18 < 6) {
                for (int var19 = 0; var19 < 3; var19 = var19 + 1) {
                  int var21 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var18 - 1) * 5)) + (((var16 * 1) + (var19 * 1)) - 1)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var17 * 3)) + var19;
                  var2.data[var6] = var2.data[var6] + (var21 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var22 = 4; var22 < 5; var22 = var22 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var22;
            for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
              int var24 = (var9 * 1) + (var23 * 1);
              if (var24 < 1) {
                continue;
              } 
              if (var24 < 6) {
                for (int var25 = 0; var25 < 3; var25 = var25 + 1) {
                  int var26 = (var22 * 1) + (var25 * 1);
                  if (var26 < 1) {
                    continue;
                  } 
                  if (var26 < 6) {
                    int var27 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var24 - 1) * 5)) + (var26 - 1)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var23 * 3)) + var25;
                    var2.data[var6] = var2.data[var6] + (var27 * arg1[var7]);
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
        // looping over the output
        for (int var28 = 1; var28 < 4; var28 = var28 + 1) {
          for (int var29 = 0; var29 < 1; var29 = var29 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var29;
            for (int var30 = 0; var30 < 3; var30 = var30 + 1) {
              int var31 = (var28 * 1) + (var30 * 1);
              for (int var32 = 0; var32 < 3; var32 = var32 + 1) {
                int var33 = (var29 * 1) + (var32 * 1);
                if (var33 < 1) {
                  continue;
                } 
                if (var33 < 6) {
                  int var34 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var31 - 1) * 5)) + (var33 - 1)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var30 * 3)) + var32;
                  var2.data[var6] = var2.data[var6] + (var34 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var35 = 1; var35 < 4; var35 = var35 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var35;
            for (int var36 = 0; var36 < 3; var36 = var36 + 1) {
              int var37 = (var28 * 1) + (var36 * 1);
              for (int var38 = 0; var38 < 3; var38 = var38 + 1) {
                int var40 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var37 - 1) * 5)) + (((var35 * 1) + (var38 * 1)) - 1)];
                var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var36 * 3)) + var38;
                var2.data[var6] = var2.data[var6] + (var40 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var41 = 4; var41 < 5; var41 = var41 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var41;
            for (int var42 = 0; var42 < 3; var42 = var42 + 1) {
              int var43 = (var28 * 1) + (var42 * 1);
              for (int var44 = 0; var44 < 3; var44 = var44 + 1) {
                int var45 = (var41 * 1) + (var44 * 1);
                if (var45 < 1) {
                  continue;
                } 
                if (var45 < 6) {
                  int var46 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var43 - 1) * 5)) + (var45 - 1)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var42 * 3)) + var44;
                  var2.data[var6] = var2.data[var6] + (var46 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var47 = 4; var47 < 5; var47 = var47 + 1) {
          for (int var48 = 0; var48 < 1; var48 = var48 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var48;
            for (int var49 = 0; var49 < 3; var49 = var49 + 1) {
              int var50 = (var47 * 1) + (var49 * 1);
              if (var50 < 1) {
                continue;
              } 
              if (var50 < 6) {
                for (int var51 = 0; var51 < 3; var51 = var51 + 1) {
                  int var52 = (var48 * 1) + (var51 * 1);
                  if (var52 < 1) {
                    continue;
                  } 
                  if (var52 < 6) {
                    int var53 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var50 - 1) * 5)) + (var52 - 1)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var49 * 3)) + var51;
                    var2.data[var6] = var2.data[var6] + (var53 * arg1[var7]);
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
          for (int var54 = 1; var54 < 4; var54 = var54 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var54;
            for (int var55 = 0; var55 < 3; var55 = var55 + 1) {
              int var56 = (var47 * 1) + (var55 * 1);
              if (var56 < 1) {
                continue;
              } 
              if (var56 < 6) {
                for (int var57 = 0; var57 < 3; var57 = var57 + 1) {
                  int var59 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var56 - 1) * 5)) + (((var54 * 1) + (var57 * 1)) - 1)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var55 * 3)) + var57;
                  var2.data[var6] = var2.data[var6] + (var59 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var60 = 4; var60 < 5; var60 = var60 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var60;
            for (int var61 = 0; var61 < 3; var61 = var61 + 1) {
              int var62 = (var47 * 1) + (var61 * 1);
              if (var62 < 1) {
                continue;
              } 
              if (var62 < 6) {
                for (int var63 = 0; var63 < 3; var63 = var63 + 1) {
                  int var64 = (var60 * 1) + (var63 * 1);
                  if (var64 < 1) {
                    continue;
                  } 
                  if (var64 < 6) {
                    int var65 = arg0[(((((var3 * 1) * 5) * 5) + ((var5 * 5) * 5)) + ((var62 - 1) * 5)) + (var64 - 1)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var61 * 3)) + var63;
                    var2.data[var6] = var2.data[var6] + (var65 * arg1[var7]);
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
  var2.mult_cnt = ((var2.mult_cnt * 1) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 2; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 1; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var12 - 3) * 15)) + (var14 - 4)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var11 * 2)) + var13;
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
          for (int var16 = 1; var16 < 2; var16 = var16 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var16;
            for (int var17 = 0; var17 < 3; var17 = var17 + 1) {
              int var18 = (var9 * 2) + (var17 * 3);
              if (var18 < 3) {
                continue;
              } 
              if (var18 < 23) {
                for (int var19 = 0; var19 < 2; var19 = var19 + 1) {
                  int var20 = (var16 * 3) + (var19 * 2);
                  if (var20 < 4) {
                    continue;
                  } 
                  if (var20 < 19) {
                    int var21 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var18 - 3) * 15)) + (var20 - 4)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var17 * 2)) + var19;
                    var2.data[var6] = var2.data[var6] + (var21 * arg1[var7]);
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
          for (int var22 = 2; var22 < 6; var22 = var22 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var22;
            for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
              int var24 = (var9 * 2) + (var23 * 3);
              if (var24 < 3) {
                continue;
              } 
              if (var24 < 23) {
                for (int var25 = 0; var25 < 2; var25 = var25 + 1) {
                  int var27 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var24 - 3) * 15)) + (((var22 * 3) + (var25 * 2)) - 4)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var23 * 2)) + var25;
                  var2.data[var6] = var2.data[var6] + (var27 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var28 = 6; var28 < 7; var28 = var28 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var28;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = (var9 * 2) + (var29 * 3);
              if (var30 < 3) {
                continue;
              } 
              if (var30 < 23) {
                for (int var31 = 0; var31 < 2; var31 = var31 + 1) {
                  int var32 = (var28 * 3) + (var31 * 2);
                  if (var32 < 4) {
                    continue;
                  } 
                  if (var32 < 19) {
                    int var33 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var30 - 3) * 15)) + (var32 - 4)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var29 * 2)) + var31;
                    var2.data[var6] = var2.data[var6] + (var33 * arg1[var7]);
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
        // looping over the output
        for (int var34 = 2; var34 < 9; var34 = var34 + 1) {
          for (int var35 = 0; var35 < 1; var35 = var35 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var35;
            for (int var36 = 0; var36 < 3; var36 = var36 + 1) {
              int var37 = (var34 * 2) + (var36 * 3);
              for (int var38 = 0; var38 < 2; var38 = var38 + 1) {
                int var39 = (var35 * 3) + (var38 * 2);
                if (var39 < 4) {
                  continue;
                } 
                if (var39 < 19) {
                  int var40 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var37 - 3) * 15)) + (var39 - 4)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var36 * 2)) + var38;
                  var2.data[var6] = var2.data[var6] + (var40 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var41 = 1; var41 < 2; var41 = var41 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var41;
            for (int var42 = 0; var42 < 3; var42 = var42 + 1) {
              int var43 = (var34 * 2) + (var42 * 3);
              for (int var44 = 0; var44 < 2; var44 = var44 + 1) {
                int var45 = (var41 * 3) + (var44 * 2);
                if (var45 < 4) {
                  continue;
                } 
                if (var45 < 19) {
                  int var46 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var43 - 3) * 15)) + (var45 - 4)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var42 * 2)) + var44;
                  var2.data[var6] = var2.data[var6] + (var46 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var47 = 2; var47 < 6; var47 = var47 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var47;
            for (int var48 = 0; var48 < 3; var48 = var48 + 1) {
              int var49 = (var34 * 2) + (var48 * 3);
              for (int var50 = 0; var50 < 2; var50 = var50 + 1) {
                int var52 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var49 - 3) * 15)) + (((var47 * 3) + (var50 * 2)) - 4)];
                var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var48 * 2)) + var50;
                var2.data[var6] = var2.data[var6] + (var52 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var53 = 6; var53 < 7; var53 = var53 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var53;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = (var34 * 2) + (var54 * 3);
              for (int var56 = 0; var56 < 2; var56 = var56 + 1) {
                int var57 = (var53 * 3) + (var56 * 2);
                if (var57 < 4) {
                  continue;
                } 
                if (var57 < 19) {
                  int var58 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var55 - 3) * 15)) + (var57 - 4)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var54 * 2)) + var56;
                  var2.data[var6] = var2.data[var6] + (var58 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var59 = 9; var59 < 10; var59 = var59 + 1) {
          for (int var60 = 0; var60 < 1; var60 = var60 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var60;
            for (int var61 = 0; var61 < 3; var61 = var61 + 1) {
              int var62 = (var59 * 2) + (var61 * 3);
              if (var62 < 3) {
                continue;
              } 
              if (var62 < 23) {
                for (int var63 = 0; var63 < 2; var63 = var63 + 1) {
                  int var64 = (var60 * 3) + (var63 * 2);
                  if (var64 < 4) {
                    continue;
                  } 
                  if (var64 < 19) {
                    int var65 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var62 - 3) * 15)) + (var64 - 4)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var61 * 2)) + var63;
                    var2.data[var6] = var2.data[var6] + (var65 * arg1[var7]);
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
          for (int var66 = 1; var66 < 2; var66 = var66 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var66;
            for (int var67 = 0; var67 < 3; var67 = var67 + 1) {
              int var68 = (var59 * 2) + (var67 * 3);
              if (var68 < 3) {
                continue;
              } 
              if (var68 < 23) {
                for (int var69 = 0; var69 < 2; var69 = var69 + 1) {
                  int var70 = (var66 * 3) + (var69 * 2);
                  if (var70 < 4) {
                    continue;
                  } 
                  if (var70 < 19) {
                    int var71 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var68 - 3) * 15)) + (var70 - 4)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var67 * 2)) + var69;
                    var2.data[var6] = var2.data[var6] + (var71 * arg1[var7]);
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
          for (int var72 = 2; var72 < 6; var72 = var72 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var72;
            for (int var73 = 0; var73 < 3; var73 = var73 + 1) {
              int var74 = (var59 * 2) + (var73 * 3);
              if (var74 < 3) {
                continue;
              } 
              if (var74 < 23) {
                for (int var75 = 0; var75 < 2; var75 = var75 + 1) {
                  int var77 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var74 - 3) * 15)) + (((var72 * 3) + (var75 * 2)) - 4)];
                  var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var73 * 2)) + var75;
                  var2.data[var6] = var2.data[var6] + (var77 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var78 = 6; var78 < 7; var78 = var78 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var78;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = (var59 * 2) + (var79 * 3);
              if (var80 < 3) {
                continue;
              } 
              if (var80 < 23) {
                for (int var81 = 0; var81 < 2; var81 = var81 + 1) {
                  int var82 = (var78 * 3) + (var81 * 2);
                  if (var82 < 4) {
                    continue;
                  } 
                  if (var82 < 19) {
                    int var83 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var80 - 3) * 15)) + (var82 - 4)];
                    var7 = (((((var4 * 1) * 2) * 3) + ((var5 * 2) * 3)) + (var79 * 2)) + var81;
                    var2.data[var6] = var2.data[var6] + (var83 * arg1[var7]);
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
  var2.mult_cnt = ((var2.mult_cnt * 1) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 3; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 2; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var12 - 3) * 15)) + (var14 - 2)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var11 * 3)) + var13;
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
          for (int var16 = 2; var16 < 13; var16 = var16 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var16;
            for (int var17 = 0; var17 < 3; var17 = var17 + 1) {
              int var18 = (var9 * 1) + (var17 * 3);
              if (var18 < 3) {
                continue;
              } 
              if (var18 < 23) {
                for (int var19 = 0; var19 < 3; var19 = var19 + 1) {
                  int var21 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var18 - 3) * 15)) + (((var16 * 1) + (var19 * 2)) - 2)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var17 * 3)) + var19;
                  var2.data[var6] = var2.data[var6] + (var21 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var22 = 13; var22 < 15; var22 = var22 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var22;
            for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
              int var24 = (var9 * 1) + (var23 * 3);
              if (var24 < 3) {
                continue;
              } 
              if (var24 < 23) {
                for (int var25 = 0; var25 < 3; var25 = var25 + 1) {
                  int var26 = (var22 * 1) + (var25 * 2);
                  if (var26 < 2) {
                    continue;
                  } 
                  if (var26 < 17) {
                    int var27 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var24 - 3) * 15)) + (var26 - 2)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var23 * 3)) + var25;
                    var2.data[var6] = var2.data[var6] + (var27 * arg1[var7]);
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
        // looping over the output
        for (int var28 = 3; var28 < 17; var28 = var28 + 1) {
          for (int var29 = 0; var29 < 2; var29 = var29 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var29;
            for (int var30 = 0; var30 < 3; var30 = var30 + 1) {
              int var31 = (var28 * 1) + (var30 * 3);
              for (int var32 = 0; var32 < 3; var32 = var32 + 1) {
                int var33 = (var29 * 1) + (var32 * 2);
                if (var33 < 2) {
                  continue;
                } 
                if (var33 < 17) {
                  int var34 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var31 - 3) * 15)) + (var33 - 2)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var30 * 3)) + var32;
                  var2.data[var6] = var2.data[var6] + (var34 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var35 = 2; var35 < 13; var35 = var35 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var35;
            for (int var36 = 0; var36 < 3; var36 = var36 + 1) {
              int var37 = (var28 * 1) + (var36 * 3);
              for (int var38 = 0; var38 < 3; var38 = var38 + 1) {
                int var40 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var37 - 3) * 15)) + (((var35 * 1) + (var38 * 2)) - 2)];
                var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var36 * 3)) + var38;
                var2.data[var6] = var2.data[var6] + (var40 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var41 = 13; var41 < 15; var41 = var41 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var41;
            for (int var42 = 0; var42 < 3; var42 = var42 + 1) {
              int var43 = (var28 * 1) + (var42 * 3);
              for (int var44 = 0; var44 < 3; var44 = var44 + 1) {
                int var45 = (var41 * 1) + (var44 * 2);
                if (var45 < 2) {
                  continue;
                } 
                if (var45 < 17) {
                  int var46 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var43 - 3) * 15)) + (var45 - 2)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var42 * 3)) + var44;
                  var2.data[var6] = var2.data[var6] + (var46 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var47 = 17; var47 < 20; var47 = var47 + 1) {
          for (int var48 = 0; var48 < 2; var48 = var48 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var48;
            for (int var49 = 0; var49 < 3; var49 = var49 + 1) {
              int var50 = (var47 * 1) + (var49 * 3);
              if (var50 < 3) {
                continue;
              } 
              if (var50 < 23) {
                for (int var51 = 0; var51 < 3; var51 = var51 + 1) {
                  int var52 = (var48 * 1) + (var51 * 2);
                  if (var52 < 2) {
                    continue;
                  } 
                  if (var52 < 17) {
                    int var53 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var50 - 3) * 15)) + (var52 - 2)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var49 * 3)) + var51;
                    var2.data[var6] = var2.data[var6] + (var53 * arg1[var7]);
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
          for (int var54 = 2; var54 < 13; var54 = var54 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var54;
            for (int var55 = 0; var55 < 3; var55 = var55 + 1) {
              int var56 = (var47 * 1) + (var55 * 3);
              if (var56 < 3) {
                continue;
              } 
              if (var56 < 23) {
                for (int var57 = 0; var57 < 3; var57 = var57 + 1) {
                  int var59 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var56 - 3) * 15)) + (((var54 * 1) + (var57 * 2)) - 2)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var55 * 3)) + var57;
                  var2.data[var6] = var2.data[var6] + (var59 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var60 = 13; var60 < 15; var60 = var60 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var60;
            for (int var61 = 0; var61 < 3; var61 = var61 + 1) {
              int var62 = (var47 * 1) + (var61 * 3);
              if (var62 < 3) {
                continue;
              } 
              if (var62 < 23) {
                for (int var63 = 0; var63 < 3; var63 = var63 + 1) {
                  int var64 = (var60 * 1) + (var63 * 2);
                  if (var64 < 2) {
                    continue;
                  } 
                  if (var64 < 17) {
                    int var65 = arg0[(((((var3 * 1) * 15) * 20) + ((var5 * 15) * 20)) + ((var62 - 3) * 15)) + (var64 - 2)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var61 * 3)) + var63;
                    var2.data[var6] = var2.data[var6] + (var65 * arg1[var7]);
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
  var2.mult_cnt = ((var2.mult_cnt * 1) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 5; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 1; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 1; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 1; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var12 - 5) * 20)) + (var14 - 4)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var11 * 3)) + var13;
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
          for (int var16 = 1; var16 < 5; var16 = var16 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var16;
            for (int var17 = 0; var17 < 3; var17 = var17 + 1) {
              int var18 = (var9 * 2) + (var17 * 2);
              if (var18 < 5) {
                continue;
              } 
              if (var18 < 25) {
                for (int var19 = 0; var19 < 3; var19 = var19 + 1) {
                  int var21 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var18 - 5) * 20)) + (((var16 * 4) + (var19 * 2)) - 4)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var17 * 3)) + var19;
                  var2.data[var6] = var2.data[var6] + (var21 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var22 = 5; var22 < 6; var22 = var22 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var22;
            for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
              int var24 = (var9 * 2) + (var23 * 2);
              if (var24 < 5) {
                continue;
              } 
              if (var24 < 25) {
                for (int var25 = 0; var25 < 3; var25 = var25 + 1) {
                  int var26 = (var22 * 4) + (var25 * 2);
                  if (var26 < 4) {
                    continue;
                  } 
                  if (var26 < 24) {
                    int var27 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var24 - 5) * 20)) + (var26 - 4)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var23 * 3)) + var25;
                    var2.data[var6] = var2.data[var6] + (var27 * arg1[var7]);
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
        // looping over the output
        for (int var28 = 1; var28 < 3; var28 = var28 + 1) {
          for (int var29 = 0; var29 < 1; var29 = var29 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var29;
            for (int var30 = 0; var30 < 3; var30 = var30 + 1) {
              int var31 = (var28 * 2) + (var30 * 2);
              if (var31 < 5) {
                continue;
              } 
              if (var31 < 25) {
                for (int var32 = 0; var32 < 3; var32 = var32 + 1) {
                  int var33 = (var29 * 4) + (var32 * 2);
                  if (var33 < 4) {
                    continue;
                  } 
                  if (var33 < 24) {
                    int var34 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var31 - 5) * 20)) + (var33 - 4)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var30 * 3)) + var32;
                    var2.data[var6] = var2.data[var6] + (var34 * arg1[var7]);
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
          for (int var35 = 1; var35 < 5; var35 = var35 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var35;
            for (int var36 = 0; var36 < 3; var36 = var36 + 1) {
              int var37 = (var28 * 2) + (var36 * 2);
              if (var37 < 5) {
                continue;
              } 
              if (var37 < 25) {
                for (int var38 = 0; var38 < 3; var38 = var38 + 1) {
                  int var40 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var37 - 5) * 20)) + (((var35 * 4) + (var38 * 2)) - 4)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var36 * 3)) + var38;
                  var2.data[var6] = var2.data[var6] + (var40 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var41 = 5; var41 < 6; var41 = var41 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var41;
            for (int var42 = 0; var42 < 3; var42 = var42 + 1) {
              int var43 = (var28 * 2) + (var42 * 2);
              if (var43 < 5) {
                continue;
              } 
              if (var43 < 25) {
                for (int var44 = 0; var44 < 3; var44 = var44 + 1) {
                  int var45 = (var41 * 4) + (var44 * 2);
                  if (var45 < 4) {
                    continue;
                  } 
                  if (var45 < 24) {
                    int var46 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var43 - 5) * 20)) + (var45 - 4)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var42 * 3)) + var44;
                    var2.data[var6] = var2.data[var6] + (var46 * arg1[var7]);
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
        // looping over the output
        for (int var47 = 3; var47 < 11; var47 = var47 + 1) {
          for (int var48 = 0; var48 < 1; var48 = var48 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var48;
            for (int var49 = 0; var49 < 3; var49 = var49 + 1) {
              int var50 = (var47 * 2) + (var49 * 2);
              for (int var51 = 0; var51 < 3; var51 = var51 + 1) {
                int var52 = (var48 * 4) + (var51 * 2);
                if (var52 < 4) {
                  continue;
                } 
                if (var52 < 24) {
                  int var53 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var50 - 5) * 20)) + (var52 - 4)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var49 * 3)) + var51;
                  var2.data[var6] = var2.data[var6] + (var53 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var54 = 1; var54 < 5; var54 = var54 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var54;
            for (int var55 = 0; var55 < 3; var55 = var55 + 1) {
              int var56 = (var47 * 2) + (var55 * 2);
              for (int var57 = 0; var57 < 3; var57 = var57 + 1) {
                int var59 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var56 - 5) * 20)) + (((var54 * 4) + (var57 * 2)) - 4)];
                var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var55 * 3)) + var57;
                var2.data[var6] = var2.data[var6] + (var59 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var60 = 5; var60 < 6; var60 = var60 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var60;
            for (int var61 = 0; var61 < 3; var61 = var61 + 1) {
              int var62 = (var47 * 2) + (var61 * 2);
              for (int var63 = 0; var63 < 3; var63 = var63 + 1) {
                int var64 = (var60 * 4) + (var63 * 2);
                if (var64 < 4) {
                  continue;
                } 
                if (var64 < 24) {
                  int var65 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var62 - 5) * 20)) + (var64 - 4)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var61 * 3)) + var63;
                  var2.data[var6] = var2.data[var6] + (var65 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var66 = 11; var66 < 13; var66 = var66 + 1) {
          for (int var67 = 0; var67 < 1; var67 = var67 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var66 * var2.width)) + var67;
            for (int var68 = 0; var68 < 3; var68 = var68 + 1) {
              int var69 = (var66 * 2) + (var68 * 2);
              if (var69 < 5) {
                continue;
              } 
              if (var69 < 25) {
                for (int var70 = 0; var70 < 3; var70 = var70 + 1) {
                  int var71 = (var67 * 4) + (var70 * 2);
                  if (var71 < 4) {
                    continue;
                  } 
                  if (var71 < 24) {
                    int var72 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var69 - 5) * 20)) + (var71 - 4)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var68 * 3)) + var70;
                    var2.data[var6] = var2.data[var6] + (var72 * arg1[var7]);
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
          for (int var73 = 1; var73 < 5; var73 = var73 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var66 * var2.width)) + var73;
            for (int var74 = 0; var74 < 3; var74 = var74 + 1) {
              int var75 = (var66 * 2) + (var74 * 2);
              if (var75 < 5) {
                continue;
              } 
              if (var75 < 25) {
                for (int var76 = 0; var76 < 3; var76 = var76 + 1) {
                  int var78 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var75 - 5) * 20)) + (((var73 * 4) + (var76 * 2)) - 4)];
                  var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var74 * 3)) + var76;
                  var2.data[var6] = var2.data[var6] + (var78 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var79 = 5; var79 < 6; var79 = var79 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var66 * var2.width)) + var79;
            for (int var80 = 0; var80 < 3; var80 = var80 + 1) {
              int var81 = (var66 * 2) + (var80 * 2);
              if (var81 < 5) {
                continue;
              } 
              if (var81 < 25) {
                for (int var82 = 0; var82 < 3; var82 = var82 + 1) {
                  int var83 = (var79 * 4) + (var82 * 2);
                  if (var83 < 4) {
                    continue;
                  } 
                  if (var83 < 24) {
                    int var84 = arg0[(((((var3 * 1) * 20) * 20) + ((var5 * 20) * 20)) + ((var81 - 5) * 20)) + (var83 - 4)];
                    var7 = (((((var4 * 1) * 3) * 3) + ((var5 * 3) * 3)) + (var80 * 3)) + var82;
                    var2.data[var6] = var2.data[var6] + (var84 * arg1[var7]);
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
  var2.mult_cnt = ((var2.mult_cnt * 5) * 1) * 1;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 4; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 5; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 3; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 3; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 1; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
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
                    int var15 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var12 - 5) * 20)) + (var14 - 4)];
                    var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var11 * 5)) + var13;
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
          for (int var16 = 1; var16 < 4; var16 = var16 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var16;
            for (int var17 = 0; var17 < 5; var17 = var17 + 1) {
              int var18 = (var9 * 2) + (var17 * 2);
              if (var18 < 5) {
                continue;
              } 
              if (var18 < 25) {
                for (int var19 = 0; var19 < 5; var19 = var19 + 1) {
                  int var21 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var18 - 5) * 20)) + (((var16 * 4) + (var19 * 2)) - 4)];
                  var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var17 * 5)) + var19;
                  var2.data[var6] = var2.data[var6] + (var21 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var22 = 4; var22 < 5; var22 = var22 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var22;
            for (int var23 = 0; var23 < 5; var23 = var23 + 1) {
              int var24 = (var9 * 2) + (var23 * 2);
              if (var24 < 5) {
                continue;
              } 
              if (var24 < 25) {
                for (int var25 = 0; var25 < 5; var25 = var25 + 1) {
                  int var26 = (var22 * 4) + (var25 * 2);
                  if (var26 < 4) {
                    continue;
                  } 
                  if (var26 < 24) {
                    int var27 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var24 - 5) * 20)) + (var26 - 4)];
                    var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var23 * 5)) + var25;
                    var2.data[var6] = var2.data[var6] + (var27 * arg1[var7]);
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
        // looping over the output
        for (int var28 = 3; var28 < 9; var28 = var28 + 1) {
          for (int var29 = 0; var29 < 1; var29 = var29 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var29;
            for (int var30 = 0; var30 < 5; var30 = var30 + 1) {
              int var31 = (var28 * 2) + (var30 * 2);
              for (int var32 = 0; var32 < 5; var32 = var32 + 1) {
                int var33 = (var29 * 4) + (var32 * 2);
                if (var33 < 4) {
                  continue;
                } 
                if (var33 < 24) {
                  int var34 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var31 - 5) * 20)) + (var33 - 4)];
                  var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var30 * 5)) + var32;
                  var2.data[var6] = var2.data[var6] + (var34 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var35 = 1; var35 < 4; var35 = var35 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var35;
            for (int var36 = 0; var36 < 5; var36 = var36 + 1) {
              int var37 = (var28 * 2) + (var36 * 2);
              for (int var38 = 0; var38 < 5; var38 = var38 + 1) {
                int var40 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var37 - 5) * 20)) + (((var35 * 4) + (var38 * 2)) - 4)];
                var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var36 * 5)) + var38;
                var2.data[var6] = var2.data[var6] + (var40 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var41 = 4; var41 < 5; var41 = var41 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var28 * var2.width)) + var41;
            for (int var42 = 0; var42 < 5; var42 = var42 + 1) {
              int var43 = (var28 * 2) + (var42 * 2);
              for (int var44 = 0; var44 < 5; var44 = var44 + 1) {
                int var45 = (var41 * 4) + (var44 * 2);
                if (var45 < 4) {
                  continue;
                } 
                if (var45 < 24) {
                  int var46 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var43 - 5) * 20)) + (var45 - 4)];
                  var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var42 * 5)) + var44;
                  var2.data[var6] = var2.data[var6] + (var46 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var47 = 9; var47 < 11; var47 = var47 + 1) {
          for (int var48 = 0; var48 < 1; var48 = var48 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var48;
            for (int var49 = 0; var49 < 5; var49 = var49 + 1) {
              int var50 = (var47 * 2) + (var49 * 2);
              if (var50 < 5) {
                continue;
              } 
              if (var50 < 25) {
                for (int var51 = 0; var51 < 5; var51 = var51 + 1) {
                  int var52 = (var48 * 4) + (var51 * 2);
                  if (var52 < 4) {
                    continue;
                  } 
                  if (var52 < 24) {
                    int var53 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var50 - 5) * 20)) + (var52 - 4)];
                    var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var49 * 5)) + var51;
                    var2.data[var6] = var2.data[var6] + (var53 * arg1[var7]);
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
          for (int var54 = 1; var54 < 4; var54 = var54 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var54;
            for (int var55 = 0; var55 < 5; var55 = var55 + 1) {
              int var56 = (var47 * 2) + (var55 * 2);
              if (var56 < 5) {
                continue;
              } 
              if (var56 < 25) {
                for (int var57 = 0; var57 < 5; var57 = var57 + 1) {
                  int var59 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var56 - 5) * 20)) + (((var54 * 4) + (var57 * 2)) - 4)];
                  var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var55 * 5)) + var57;
                  var2.data[var6] = var2.data[var6] + (var59 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var60 = 4; var60 < 5; var60 = var60 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var47 * var2.width)) + var60;
            for (int var61 = 0; var61 < 5; var61 = var61 + 1) {
              int var62 = (var47 * 2) + (var61 * 2);
              if (var62 < 5) {
                continue;
              } 
              if (var62 < 25) {
                for (int var63 = 0; var63 < 5; var63 = var63 + 1) {
                  int var64 = (var60 * 4) + (var63 * 2);
                  if (var64 < 4) {
                    continue;
                  } 
                  if (var64 < 24) {
                    int var65 = arg0[(((((var3 * 3) * 20) * 20) + ((var5 * 20) * 20)) + ((var62 - 5) * 20)) + (var64 - 4)];
                    var7 = (((((var4 * 3) * 5) * 5) + ((var5 * 5) * 5)) + (var61 * 5)) + var63;
                    var2.data[var6] = var2.data[var6] + (var65 * arg1[var7]);
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
  var2.mult_cnt = ((var2.mult_cnt * 4) * 3) * 5;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 10; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 10; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 91; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 91; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                int var15 = arg0[(((((var3 * 10) * 100) * 100) + ((var5 * 100) * 100)) + ((var12 - 0) * 100)) + (((var10 * 1) + (var13 * 1)) - 0)];
                var7 = (((((var4 * 10) * 10) * 10) + ((var5 * 10) * 10)) + (var11 * 10)) + var13;
                var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                var8 = var8 + 1;
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



conv_runtime::ImageT<int> conv2d_stride4x4_im100x100_w10x10_batch10_inch5_outch10 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 23;
  var2.width = 23;
  var2.in_channels = 10;
  var2.batch_size = 10;
  var2.data = conv_runtime::conv_calloc(52900, 4);
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 10; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 5; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 23; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 23; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            for (int var11 = 0; var11 < 10; var11 = var11 + 1) {
              int var12 = (var9 * 4) + (var11 * 1);
              for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
                int var15 = arg0[(((((var3 * 5) * 100) * 100) + ((var5 * 100) * 100)) + ((var12 - 0) * 100)) + (((var10 * 4) + (var13 * 1)) - 0)];
                var7 = (((((var4 * 5) * 10) * 10) + ((var5 * 10) * 10)) + (var11 * 10)) + var13;
                var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 10) * 5) * 10;
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
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 10; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 1; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 5; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 6; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 6; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            for (int var11 = 0; var11 < 5; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 1);
              for (int var13 = 0; var13 < 5; var13 = var13 + 1) {
                int var15 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var12 - 0) * 10)) + (((var10 * 1) + (var13 * 1)) - 0)];
                var7 = (((((var4 * 5) * 5) * 5) + ((var5 * 5) * 5)) + (var11 * 5)) + var13;
                var2.data[var6] = var2.data[var6] + (var15 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
        }
        var2.mult_cnt = var8;
      }
    }
  }
  var2.mult_cnt = ((var2.mult_cnt * 10) * 5) * 1;
  return var2;
}



conv_runtime::ImageT<int> conv2d_pad10x10_dil4x4_im10x10_w3x3_batch1_inch5_outch3 (int* arg0, int* arg1) {
  assert(1);
  assert(1);
  conv_runtime::ImageT<int> var2;
  var2.height = 22;
  var2.width = 22;
  var2.in_channels = 3;
  var2.batch_size = 1;
  var2.data = conv_runtime::conv_calloc(1452, 4);
  // looping over batches 
  #pragma  omp parallel for collapse(3)
  for (int var3 = 0; var3 < 1; var3 = var3 + 1) {
    // looping over out channels
    for (int var4 = 0; var4 < 3; var4 = var4 + 1) {
      // looping over in channels
      for (int var5 = 0; var5 < 5; var5 = var5 + 1) {
        int var6;
        int var7;
        int var8 = 0;
        // looping over the output
        for (int var9 = 0; var9 < 2; var9 = var9 + 1) {
          for (int var10 = 0; var10 < 2; var10 = var10 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var10;
            for (int var11 = 0; var11 < 3; var11 = var11 + 1) {
              int var12 = (var9 * 1) + (var11 * 4);
              if (var12 < 10) {
                continue;
              } 
              if (var12 < 20) {
                for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
                  int var14 = (var10 * 1) + (var13 * 4);
                  if (var14 < 10) {
                    continue;
                  } 
                  if (var14 < 20) {
                    int var15 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var12 - 10) * 10)) + (var14 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var11 * 3)) + var13;
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
          for (int var16 = 2; var16 < 10; var16 = var16 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var16;
            for (int var17 = 0; var17 < 3; var17 = var17 + 1) {
              int var18 = (var9 * 1) + (var17 * 4);
              if (var18 < 10) {
                continue;
              } 
              if (var18 < 20) {
                for (int var19 = 0; var19 < 3; var19 = var19 + 1) {
                  int var20 = (var16 * 1) + (var19 * 4);
                  if (var20 < 10) {
                    continue;
                  } 
                  if (var20 < 20) {
                    int var21 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var18 - 10) * 10)) + (var20 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var17 * 3)) + var19;
                    var2.data[var6] = var2.data[var6] + (var21 * arg1[var7]);
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
          for (int var22 = 10; var22 < 12; var22 = var22 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var22;
            for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
              int var24 = (var9 * 1) + (var23 * 4);
              if (var24 < 10) {
                continue;
              } 
              if (var24 < 20) {
                for (int var25 = 0; var25 < 3; var25 = var25 + 1) {
                  int var27 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var24 - 10) * 10)) + (((var22 * 1) + (var25 * 4)) - 10)];
                  var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var23 * 3)) + var25;
                  var2.data[var6] = var2.data[var6] + (var27 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var28 = 12; var28 < 22; var28 = var28 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var9 * var2.width)) + var28;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = (var9 * 1) + (var29 * 4);
              if (var30 < 10) {
                continue;
              } 
              if (var30 < 20) {
                for (int var31 = 0; var31 < 3; var31 = var31 + 1) {
                  int var32 = (var28 * 1) + (var31 * 4);
                  if (var32 < 10) {
                    continue;
                  } 
                  if (var32 < 20) {
                    int var33 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var30 - 10) * 10)) + (var32 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var29 * 3)) + var31;
                    var2.data[var6] = var2.data[var6] + (var33 * arg1[var7]);
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
        // looping over the output
        for (int var34 = 2; var34 < 10; var34 = var34 + 1) {
          for (int var35 = 0; var35 < 2; var35 = var35 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var35;
            for (int var36 = 0; var36 < 3; var36 = var36 + 1) {
              int var37 = (var34 * 1) + (var36 * 4);
              if (var37 < 10) {
                continue;
              } 
              if (var37 < 20) {
                for (int var38 = 0; var38 < 3; var38 = var38 + 1) {
                  int var39 = (var35 * 1) + (var38 * 4);
                  if (var39 < 10) {
                    continue;
                  } 
                  if (var39 < 20) {
                    int var40 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var37 - 10) * 10)) + (var39 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var36 * 3)) + var38;
                    var2.data[var6] = var2.data[var6] + (var40 * arg1[var7]);
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
          for (int var41 = 2; var41 < 10; var41 = var41 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var41;
            for (int var42 = 0; var42 < 3; var42 = var42 + 1) {
              int var43 = (var34 * 1) + (var42 * 4);
              if (var43 < 10) {
                continue;
              } 
              if (var43 < 20) {
                for (int var44 = 0; var44 < 3; var44 = var44 + 1) {
                  int var45 = (var41 * 1) + (var44 * 4);
                  if (var45 < 10) {
                    continue;
                  } 
                  if (var45 < 20) {
                    int var46 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var43 - 10) * 10)) + (var45 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var42 * 3)) + var44;
                    var2.data[var6] = var2.data[var6] + (var46 * arg1[var7]);
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
          for (int var47 = 10; var47 < 12; var47 = var47 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var47;
            for (int var48 = 0; var48 < 3; var48 = var48 + 1) {
              int var49 = (var34 * 1) + (var48 * 4);
              if (var49 < 10) {
                continue;
              } 
              if (var49 < 20) {
                for (int var50 = 0; var50 < 3; var50 = var50 + 1) {
                  int var52 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var49 - 10) * 10)) + (((var47 * 1) + (var50 * 4)) - 10)];
                  var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var48 * 3)) + var50;
                  var2.data[var6] = var2.data[var6] + (var52 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var53 = 12; var53 < 22; var53 = var53 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var34 * var2.width)) + var53;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = (var34 * 1) + (var54 * 4);
              if (var55 < 10) {
                continue;
              } 
              if (var55 < 20) {
                for (int var56 = 0; var56 < 3; var56 = var56 + 1) {
                  int var57 = (var53 * 1) + (var56 * 4);
                  if (var57 < 10) {
                    continue;
                  } 
                  if (var57 < 20) {
                    int var58 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var55 - 10) * 10)) + (var57 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var54 * 3)) + var56;
                    var2.data[var6] = var2.data[var6] + (var58 * arg1[var7]);
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
        // looping over the output
        for (int var59 = 10; var59 < 12; var59 = var59 + 1) {
          for (int var60 = 0; var60 < 2; var60 = var60 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var60;
            for (int var61 = 0; var61 < 3; var61 = var61 + 1) {
              int var62 = (var59 * 1) + (var61 * 4);
              for (int var63 = 0; var63 < 3; var63 = var63 + 1) {
                int var64 = (var60 * 1) + (var63 * 4);
                if (var64 < 10) {
                  continue;
                } 
                if (var64 < 20) {
                  int var65 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var62 - 10) * 10)) + (var64 - 10)];
                  var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var61 * 3)) + var63;
                  var2.data[var6] = var2.data[var6] + (var65 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var66 = 2; var66 < 10; var66 = var66 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var66;
            for (int var67 = 0; var67 < 3; var67 = var67 + 1) {
              int var68 = (var59 * 1) + (var67 * 4);
              for (int var69 = 0; var69 < 3; var69 = var69 + 1) {
                int var70 = (var66 * 1) + (var69 * 4);
                if (var70 < 10) {
                  continue;
                } 
                if (var70 < 20) {
                  int var71 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var68 - 10) * 10)) + (var70 - 10)];
                  var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var67 * 3)) + var69;
                  var2.data[var6] = var2.data[var6] + (var71 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var72 = 10; var72 < 12; var72 = var72 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var72;
            for (int var73 = 0; var73 < 3; var73 = var73 + 1) {
              int var74 = (var59 * 1) + (var73 * 4);
              for (int var75 = 0; var75 < 3; var75 = var75 + 1) {
                int var77 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var74 - 10) * 10)) + (((var72 * 1) + (var75 * 4)) - 10)];
                var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var73 * 3)) + var75;
                var2.data[var6] = var2.data[var6] + (var77 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var78 = 12; var78 < 22; var78 = var78 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var59 * var2.width)) + var78;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = (var59 * 1) + (var79 * 4);
              for (int var81 = 0; var81 < 3; var81 = var81 + 1) {
                int var82 = (var78 * 1) + (var81 * 4);
                if (var82 < 10) {
                  continue;
                } 
                if (var82 < 20) {
                  int var83 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var80 - 10) * 10)) + (var82 - 10)];
                  var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var79 * 3)) + var81;
                  var2.data[var6] = var2.data[var6] + (var83 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var84 = 12; var84 < 22; var84 = var84 + 1) {
          for (int var85 = 0; var85 < 2; var85 = var85 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var84 * var2.width)) + var85;
            for (int var86 = 0; var86 < 3; var86 = var86 + 1) {
              int var87 = (var84 * 1) + (var86 * 4);
              if (var87 < 10) {
                continue;
              } 
              if (var87 < 20) {
                for (int var88 = 0; var88 < 3; var88 = var88 + 1) {
                  int var89 = (var85 * 1) + (var88 * 4);
                  if (var89 < 10) {
                    continue;
                  } 
                  if (var89 < 20) {
                    int var90 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var87 - 10) * 10)) + (var89 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var86 * 3)) + var88;
                    var2.data[var6] = var2.data[var6] + (var90 * arg1[var7]);
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
          for (int var91 = 2; var91 < 10; var91 = var91 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var84 * var2.width)) + var91;
            for (int var92 = 0; var92 < 3; var92 = var92 + 1) {
              int var93 = (var84 * 1) + (var92 * 4);
              if (var93 < 10) {
                continue;
              } 
              if (var93 < 20) {
                for (int var94 = 0; var94 < 3; var94 = var94 + 1) {
                  int var95 = (var91 * 1) + (var94 * 4);
                  if (var95 < 10) {
                    continue;
                  } 
                  if (var95 < 20) {
                    int var96 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var93 - 10) * 10)) + (var95 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var92 * 3)) + var94;
                    var2.data[var6] = var2.data[var6] + (var96 * arg1[var7]);
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
          for (int var97 = 10; var97 < 12; var97 = var97 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var84 * var2.width)) + var97;
            for (int var98 = 0; var98 < 3; var98 = var98 + 1) {
              int var99 = (var84 * 1) + (var98 * 4);
              if (var99 < 10) {
                continue;
              } 
              if (var99 < 20) {
                for (int var100 = 0; var100 < 3; var100 = var100 + 1) {
                  int var102 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var99 - 10) * 10)) + (((var97 * 1) + (var100 * 4)) - 10)];
                  var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var98 * 3)) + var100;
                  var2.data[var6] = var2.data[var6] + (var102 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var103 = 12; var103 < 22; var103 = var103 + 1) {
            var6 = (((((var3 * var2.in_channels) * var2.height) * var2.width) + ((var4 * var2.width) * var2.height)) + (var84 * var2.width)) + var103;
            for (int var104 = 0; var104 < 3; var104 = var104 + 1) {
              int var105 = (var84 * 1) + (var104 * 4);
              if (var105 < 10) {
                continue;
              } 
              if (var105 < 20) {
                for (int var106 = 0; var106 < 3; var106 = var106 + 1) {
                  int var107 = (var103 * 1) + (var106 * 4);
                  if (var107 < 10) {
                    continue;
                  } 
                  if (var107 < 20) {
                    int var108 = arg0[(((((var3 * 5) * 10) * 10) + ((var5 * 10) * 10)) + ((var105 - 10) * 10)) + (var107 - 10)];
                    var7 = (((((var4 * 5) * 3) * 3) + ((var5 * 3) * 3)) + (var104 * 3)) + var106;
                    var2.data[var6] = var2.data[var6] + (var108 * arg1[var7]);
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
  var2.mult_cnt = ((var2.mult_cnt * 1) * 5) * 3;
  return var2;
}



