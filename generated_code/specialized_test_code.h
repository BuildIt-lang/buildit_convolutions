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
            var6 = (((var3 * 9) + (var4 * 9)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 1;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 1);
              for (int var15 = 0; var15 < 3; var15 = var15 + 1) {
                int var17 = arg0[(((var3 * 25) + (var5 * 25)) + ((var14 - 0) * 5)) + ((var11 + (var15 * 1)) - 0)];
                var7 = (((var4 * 9) + (var5 * 9)) + (var13 * 3)) + var15;
                var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
            var6 = (((var3 * 27) + (var4 * 27)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 2;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 1);
              for (int var15 = 0; var15 < 2; var15 = var15 + 1) {
                int var17 = arg0[(((var3 * 80) + (var5 * 80)) + ((var14 - 0) * 10)) + ((var11 + (var15 * 1)) - 0)];
                var7 = (((var4 * 6) + (var5 * 6)) + (var13 * 2)) + var15;
                var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
            var6 = (((var3 * 182) + (var4 * 182)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 1;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 3);
              for (int var15 = 0; var15 < 2; var15 = var15 + 1) {
                int var17 = arg0[(((var3 * 300) + (var5 * 300)) + ((var14 - 0) * 15)) + ((var11 + (var15 * 2)) - 0)];
                var7 = (((var4 * 6) + (var5 * 6)) + (var13 * 2)) + var15;
                var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
            var6 = (((var3 * 35) + (var4 * 35)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 3;
            int var12 = var9 * 2;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 3);
              for (int var15 = 0; var15 < 2; var15 = var15 + 1) {
                int var17 = arg0[(((var3 * 300) + (var5 * 300)) + ((var14 - 0) * 15)) + ((var11 + (var15 * 2)) - 0)];
                var7 = (((var4 * 6) + (var5 * 6)) + (var13 * 2)) + var15;
                var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
            var6 = (((var3 * 40) + (var4 * 40)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 1;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 1);
              if (var14 < 1) {
                continue;
              } 
              if (var14 < 6) {
                for (int var15 = 0; var15 < 2; var15 = var15 + 1) {
                  int var16 = var11 + (var15 * 1);
                  if (var16 < 2) {
                    continue;
                  } 
                  if (var16 < 7) {
                    int var17 = arg0[(((var3 * 25) + (var5 * 25)) + ((var14 - 1) * 5)) + (var16 - 2)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var13 * 2)) + var15;
                    var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
          for (int var18 = 1; var18 < 2; var18 = var18 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var9 * var2.width)) + var18;
            int var19 = var18 * 1;
            int var20 = var9 * 1;
            for (int var21 = 0; var21 < 3; var21 = var21 + 1) {
              int var22 = var20 + (var21 * 1);
              if (var22 < 1) {
                continue;
              } 
              if (var22 < 6) {
                for (int var23 = 0; var23 < 2; var23 = var23 + 1) {
                  int var24 = var19 + (var23 * 1);
                  if (var24 < 2) {
                    continue;
                  } 
                  if (var24 < 7) {
                    int var25 = arg0[(((var3 * 25) + (var5 * 25)) + ((var22 - 1) * 5)) + (var24 - 2)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var21 * 2)) + var23;
                    var2.data[var6] = var2.data[var6] + (var25 * arg1[var7]);
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
          for (int var26 = 2; var26 < 6; var26 = var26 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var9 * var2.width)) + var26;
            int var27 = var26 * 1;
            int var28 = var9 * 1;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = var28 + (var29 * 1);
              if (var30 < 1) {
                continue;
              } 
              if (var30 < 6) {
                for (int var31 = 0; var31 < 2; var31 = var31 + 1) {
                  int var33 = arg0[(((var3 * 25) + (var5 * 25)) + ((var30 - 1) * 5)) + ((var27 + (var31 * 1)) - 2)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var29 * 2)) + var31;
                  var2.data[var6] = var2.data[var6] + (var33 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var34 = 6; var34 < 8; var34 = var34 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var9 * var2.width)) + var34;
            int var35 = var34 * 1;
            int var36 = var9 * 1;
            for (int var37 = 0; var37 < 3; var37 = var37 + 1) {
              int var38 = var36 + (var37 * 1);
              if (var38 < 1) {
                continue;
              } 
              if (var38 < 6) {
                for (int var39 = 0; var39 < 2; var39 = var39 + 1) {
                  int var40 = var35 + (var39 * 1);
                  if (var40 < 2) {
                    continue;
                  } 
                  if (var40 < 7) {
                    int var41 = arg0[(((var3 * 25) + (var5 * 25)) + ((var38 - 1) * 5)) + (var40 - 2)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var37 * 2)) + var39;
                    var2.data[var6] = var2.data[var6] + (var41 * arg1[var7]);
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
        for (int var42 = 1; var42 < 4; var42 = var42 + 1) {
          for (int var43 = 0; var43 < 1; var43 = var43 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var42 * var2.width)) + var43;
            int var44 = var43 * 1;
            int var45 = var42 * 1;
            for (int var46 = 0; var46 < 3; var46 = var46 + 1) {
              int var47 = var45 + (var46 * 1);
              for (int var48 = 0; var48 < 2; var48 = var48 + 1) {
                int var49 = var44 + (var48 * 1);
                if (var49 < 2) {
                  continue;
                } 
                if (var49 < 7) {
                  int var50 = arg0[(((var3 * 25) + (var5 * 25)) + ((var47 - 1) * 5)) + (var49 - 2)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var46 * 2)) + var48;
                  var2.data[var6] = var2.data[var6] + (var50 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var51 = 1; var51 < 2; var51 = var51 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var42 * var2.width)) + var51;
            int var52 = var51 * 1;
            int var53 = var42 * 1;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = var53 + (var54 * 1);
              for (int var56 = 0; var56 < 2; var56 = var56 + 1) {
                int var57 = var52 + (var56 * 1);
                if (var57 < 2) {
                  continue;
                } 
                if (var57 < 7) {
                  int var58 = arg0[(((var3 * 25) + (var5 * 25)) + ((var55 - 1) * 5)) + (var57 - 2)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var54 * 2)) + var56;
                  var2.data[var6] = var2.data[var6] + (var58 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var59 = 2; var59 < 6; var59 = var59 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var42 * var2.width)) + var59;
            int var60 = var59 * 1;
            int var61 = var42 * 1;
            for (int var62 = 0; var62 < 3; var62 = var62 + 1) {
              int var63 = var61 + (var62 * 1);
              for (int var64 = 0; var64 < 2; var64 = var64 + 1) {
                int var66 = arg0[(((var3 * 25) + (var5 * 25)) + ((var63 - 1) * 5)) + ((var60 + (var64 * 1)) - 2)];
                var7 = (((var4 * 6) + (var5 * 6)) + (var62 * 2)) + var64;
                var2.data[var6] = var2.data[var6] + (var66 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var67 = 6; var67 < 8; var67 = var67 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var42 * var2.width)) + var67;
            int var68 = var67 * 1;
            int var69 = var42 * 1;
            for (int var70 = 0; var70 < 3; var70 = var70 + 1) {
              int var71 = var69 + (var70 * 1);
              for (int var72 = 0; var72 < 2; var72 = var72 + 1) {
                int var73 = var68 + (var72 * 1);
                if (var73 < 2) {
                  continue;
                } 
                if (var73 < 7) {
                  int var74 = arg0[(((var3 * 25) + (var5 * 25)) + ((var71 - 1) * 5)) + (var73 - 2)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var70 * 2)) + var72;
                  var2.data[var6] = var2.data[var6] + (var74 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var75 = 4; var75 < 5; var75 = var75 + 1) {
          for (int var76 = 0; var76 < 1; var76 = var76 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var75 * var2.width)) + var76;
            int var77 = var76 * 1;
            int var78 = var75 * 1;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = var78 + (var79 * 1);
              if (var80 < 1) {
                continue;
              } 
              if (var80 < 6) {
                for (int var81 = 0; var81 < 2; var81 = var81 + 1) {
                  int var82 = var77 + (var81 * 1);
                  if (var82 < 2) {
                    continue;
                  } 
                  if (var82 < 7) {
                    int var83 = arg0[(((var3 * 25) + (var5 * 25)) + ((var80 - 1) * 5)) + (var82 - 2)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var79 * 2)) + var81;
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
          for (int var84 = 1; var84 < 2; var84 = var84 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var75 * var2.width)) + var84;
            int var85 = var84 * 1;
            int var86 = var75 * 1;
            for (int var87 = 0; var87 < 3; var87 = var87 + 1) {
              int var88 = var86 + (var87 * 1);
              if (var88 < 1) {
                continue;
              } 
              if (var88 < 6) {
                for (int var89 = 0; var89 < 2; var89 = var89 + 1) {
                  int var90 = var85 + (var89 * 1);
                  if (var90 < 2) {
                    continue;
                  } 
                  if (var90 < 7) {
                    int var91 = arg0[(((var3 * 25) + (var5 * 25)) + ((var88 - 1) * 5)) + (var90 - 2)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var87 * 2)) + var89;
                    var2.data[var6] = var2.data[var6] + (var91 * arg1[var7]);
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
          for (int var92 = 2; var92 < 6; var92 = var92 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var75 * var2.width)) + var92;
            int var93 = var92 * 1;
            int var94 = var75 * 1;
            for (int var95 = 0; var95 < 3; var95 = var95 + 1) {
              int var96 = var94 + (var95 * 1);
              if (var96 < 1) {
                continue;
              } 
              if (var96 < 6) {
                for (int var97 = 0; var97 < 2; var97 = var97 + 1) {
                  int var99 = arg0[(((var3 * 25) + (var5 * 25)) + ((var96 - 1) * 5)) + ((var93 + (var97 * 1)) - 2)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var95 * 2)) + var97;
                  var2.data[var6] = var2.data[var6] + (var99 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var100 = 6; var100 < 8; var100 = var100 + 1) {
            var6 = (((var3 * 40) + (var4 * 40)) + (var75 * var2.width)) + var100;
            int var101 = var100 * 1;
            int var102 = var75 * 1;
            for (int var103 = 0; var103 < 3; var103 = var103 + 1) {
              int var104 = var102 + (var103 * 1);
              if (var104 < 1) {
                continue;
              } 
              if (var104 < 6) {
                for (int var105 = 0; var105 < 2; var105 = var105 + 1) {
                  int var106 = var101 + (var105 * 1);
                  if (var106 < 2) {
                    continue;
                  } 
                  if (var106 < 7) {
                    int var107 = arg0[(((var3 * 25) + (var5 * 25)) + ((var104 - 1) * 5)) + (var106 - 2)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var103 * 2)) + var105;
                    var2.data[var6] = var2.data[var6] + (var107 * arg1[var7]);
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
            var6 = (((var3 * 25) + (var4 * 25)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 1;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 1);
              if (var14 < 1) {
                continue;
              } 
              if (var14 < 6) {
                for (int var15 = 0; var15 < 3; var15 = var15 + 1) {
                  int var16 = var11 + (var15 * 1);
                  if (var16 < 1) {
                    continue;
                  } 
                  if (var16 < 6) {
                    int var17 = arg0[(((var3 * 25) + (var5 * 25)) + ((var14 - 1) * 5)) + (var16 - 1)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var13 * 3)) + var15;
                    var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
          for (int var18 = 1; var18 < 4; var18 = var18 + 1) {
            var6 = (((var3 * 25) + (var4 * 25)) + (var9 * var2.width)) + var18;
            int var19 = var18 * 1;
            int var20 = var9 * 1;
            for (int var21 = 0; var21 < 3; var21 = var21 + 1) {
              int var22 = var20 + (var21 * 1);
              if (var22 < 1) {
                continue;
              } 
              if (var22 < 6) {
                for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
                  int var25 = arg0[(((var3 * 25) + (var5 * 25)) + ((var22 - 1) * 5)) + ((var19 + (var23 * 1)) - 1)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var21 * 3)) + var23;
                  var2.data[var6] = var2.data[var6] + (var25 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var26 = 4; var26 < 5; var26 = var26 + 1) {
            var6 = (((var3 * 25) + (var4 * 25)) + (var9 * var2.width)) + var26;
            int var27 = var26 * 1;
            int var28 = var9 * 1;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = var28 + (var29 * 1);
              if (var30 < 1) {
                continue;
              } 
              if (var30 < 6) {
                for (int var31 = 0; var31 < 3; var31 = var31 + 1) {
                  int var32 = var27 + (var31 * 1);
                  if (var32 < 1) {
                    continue;
                  } 
                  if (var32 < 6) {
                    int var33 = arg0[(((var3 * 25) + (var5 * 25)) + ((var30 - 1) * 5)) + (var32 - 1)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var29 * 3)) + var31;
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
            var6 = (((var3 * 25) + (var4 * 25)) + (var34 * var2.width)) + var35;
            int var36 = var35 * 1;
            int var37 = var34 * 1;
            for (int var38 = 0; var38 < 3; var38 = var38 + 1) {
              int var39 = var37 + (var38 * 1);
              for (int var40 = 0; var40 < 3; var40 = var40 + 1) {
                int var41 = var36 + (var40 * 1);
                if (var41 < 1) {
                  continue;
                } 
                if (var41 < 6) {
                  int var42 = arg0[(((var3 * 25) + (var5 * 25)) + ((var39 - 1) * 5)) + (var41 - 1)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var38 * 3)) + var40;
                  var2.data[var6] = var2.data[var6] + (var42 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var43 = 1; var43 < 4; var43 = var43 + 1) {
            var6 = (((var3 * 25) + (var4 * 25)) + (var34 * var2.width)) + var43;
            int var44 = var43 * 1;
            int var45 = var34 * 1;
            for (int var46 = 0; var46 < 3; var46 = var46 + 1) {
              int var47 = var45 + (var46 * 1);
              for (int var48 = 0; var48 < 3; var48 = var48 + 1) {
                int var50 = arg0[(((var3 * 25) + (var5 * 25)) + ((var47 - 1) * 5)) + ((var44 + (var48 * 1)) - 1)];
                var7 = (((var4 * 9) + (var5 * 9)) + (var46 * 3)) + var48;
                var2.data[var6] = var2.data[var6] + (var50 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var51 = 4; var51 < 5; var51 = var51 + 1) {
            var6 = (((var3 * 25) + (var4 * 25)) + (var34 * var2.width)) + var51;
            int var52 = var51 * 1;
            int var53 = var34 * 1;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = var53 + (var54 * 1);
              for (int var56 = 0; var56 < 3; var56 = var56 + 1) {
                int var57 = var52 + (var56 * 1);
                if (var57 < 1) {
                  continue;
                } 
                if (var57 < 6) {
                  int var58 = arg0[(((var3 * 25) + (var5 * 25)) + ((var55 - 1) * 5)) + (var57 - 1)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var54 * 3)) + var56;
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
            var6 = (((var3 * 25) + (var4 * 25)) + (var59 * var2.width)) + var60;
            int var61 = var60 * 1;
            int var62 = var59 * 1;
            for (int var63 = 0; var63 < 3; var63 = var63 + 1) {
              int var64 = var62 + (var63 * 1);
              if (var64 < 1) {
                continue;
              } 
              if (var64 < 6) {
                for (int var65 = 0; var65 < 3; var65 = var65 + 1) {
                  int var66 = var61 + (var65 * 1);
                  if (var66 < 1) {
                    continue;
                  } 
                  if (var66 < 6) {
                    int var67 = arg0[(((var3 * 25) + (var5 * 25)) + ((var64 - 1) * 5)) + (var66 - 1)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var63 * 3)) + var65;
                    var2.data[var6] = var2.data[var6] + (var67 * arg1[var7]);
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
          for (int var68 = 1; var68 < 4; var68 = var68 + 1) {
            var6 = (((var3 * 25) + (var4 * 25)) + (var59 * var2.width)) + var68;
            int var69 = var68 * 1;
            int var70 = var59 * 1;
            for (int var71 = 0; var71 < 3; var71 = var71 + 1) {
              int var72 = var70 + (var71 * 1);
              if (var72 < 1) {
                continue;
              } 
              if (var72 < 6) {
                for (int var73 = 0; var73 < 3; var73 = var73 + 1) {
                  int var75 = arg0[(((var3 * 25) + (var5 * 25)) + ((var72 - 1) * 5)) + ((var69 + (var73 * 1)) - 1)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var71 * 3)) + var73;
                  var2.data[var6] = var2.data[var6] + (var75 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var76 = 4; var76 < 5; var76 = var76 + 1) {
            var6 = (((var3 * 25) + (var4 * 25)) + (var59 * var2.width)) + var76;
            int var77 = var76 * 1;
            int var78 = var59 * 1;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = var78 + (var79 * 1);
              if (var80 < 1) {
                continue;
              } 
              if (var80 < 6) {
                for (int var81 = 0; var81 < 3; var81 = var81 + 1) {
                  int var82 = var77 + (var81 * 1);
                  if (var82 < 1) {
                    continue;
                  } 
                  if (var82 < 6) {
                    int var83 = arg0[(((var3 * 25) + (var5 * 25)) + ((var80 - 1) * 5)) + (var82 - 1)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var79 * 3)) + var81;
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
            var6 = (((var3 * 70) + (var4 * 70)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 3;
            int var12 = var9 * 2;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 3);
              if (var14 < 3) {
                continue;
              } 
              if (var14 < 23) {
                for (int var15 = 0; var15 < 2; var15 = var15 + 1) {
                  int var16 = var11 + (var15 * 2);
                  if (var16 < 4) {
                    continue;
                  } 
                  if (var16 < 19) {
                    int var17 = arg0[(((var3 * 300) + (var5 * 300)) + ((var14 - 3) * 15)) + (var16 - 4)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var13 * 2)) + var15;
                    var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
          for (int var18 = 1; var18 < 2; var18 = var18 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var9 * var2.width)) + var18;
            int var19 = var18 * 3;
            int var20 = var9 * 2;
            for (int var21 = 0; var21 < 3; var21 = var21 + 1) {
              int var22 = var20 + (var21 * 3);
              if (var22 < 3) {
                continue;
              } 
              if (var22 < 23) {
                for (int var23 = 0; var23 < 2; var23 = var23 + 1) {
                  int var24 = var19 + (var23 * 2);
                  if (var24 < 4) {
                    continue;
                  } 
                  if (var24 < 19) {
                    int var25 = arg0[(((var3 * 300) + (var5 * 300)) + ((var22 - 3) * 15)) + (var24 - 4)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var21 * 2)) + var23;
                    var2.data[var6] = var2.data[var6] + (var25 * arg1[var7]);
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
          for (int var26 = 2; var26 < 6; var26 = var26 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var9 * var2.width)) + var26;
            int var27 = var26 * 3;
            int var28 = var9 * 2;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = var28 + (var29 * 3);
              if (var30 < 3) {
                continue;
              } 
              if (var30 < 23) {
                for (int var31 = 0; var31 < 2; var31 = var31 + 1) {
                  int var33 = arg0[(((var3 * 300) + (var5 * 300)) + ((var30 - 3) * 15)) + ((var27 + (var31 * 2)) - 4)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var29 * 2)) + var31;
                  var2.data[var6] = var2.data[var6] + (var33 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var34 = 6; var34 < 7; var34 = var34 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var9 * var2.width)) + var34;
            int var35 = var34 * 3;
            int var36 = var9 * 2;
            for (int var37 = 0; var37 < 3; var37 = var37 + 1) {
              int var38 = var36 + (var37 * 3);
              if (var38 < 3) {
                continue;
              } 
              if (var38 < 23) {
                for (int var39 = 0; var39 < 2; var39 = var39 + 1) {
                  int var40 = var35 + (var39 * 2);
                  if (var40 < 4) {
                    continue;
                  } 
                  if (var40 < 19) {
                    int var41 = arg0[(((var3 * 300) + (var5 * 300)) + ((var38 - 3) * 15)) + (var40 - 4)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var37 * 2)) + var39;
                    var2.data[var6] = var2.data[var6] + (var41 * arg1[var7]);
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
        for (int var42 = 2; var42 < 9; var42 = var42 + 1) {
          for (int var43 = 0; var43 < 1; var43 = var43 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var42 * var2.width)) + var43;
            int var44 = var43 * 3;
            int var45 = var42 * 2;
            for (int var46 = 0; var46 < 3; var46 = var46 + 1) {
              int var47 = var45 + (var46 * 3);
              for (int var48 = 0; var48 < 2; var48 = var48 + 1) {
                int var49 = var44 + (var48 * 2);
                if (var49 < 4) {
                  continue;
                } 
                if (var49 < 19) {
                  int var50 = arg0[(((var3 * 300) + (var5 * 300)) + ((var47 - 3) * 15)) + (var49 - 4)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var46 * 2)) + var48;
                  var2.data[var6] = var2.data[var6] + (var50 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var51 = 1; var51 < 2; var51 = var51 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var42 * var2.width)) + var51;
            int var52 = var51 * 3;
            int var53 = var42 * 2;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = var53 + (var54 * 3);
              for (int var56 = 0; var56 < 2; var56 = var56 + 1) {
                int var57 = var52 + (var56 * 2);
                if (var57 < 4) {
                  continue;
                } 
                if (var57 < 19) {
                  int var58 = arg0[(((var3 * 300) + (var5 * 300)) + ((var55 - 3) * 15)) + (var57 - 4)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var54 * 2)) + var56;
                  var2.data[var6] = var2.data[var6] + (var58 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var59 = 2; var59 < 6; var59 = var59 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var42 * var2.width)) + var59;
            int var60 = var59 * 3;
            int var61 = var42 * 2;
            for (int var62 = 0; var62 < 3; var62 = var62 + 1) {
              int var63 = var61 + (var62 * 3);
              for (int var64 = 0; var64 < 2; var64 = var64 + 1) {
                int var66 = arg0[(((var3 * 300) + (var5 * 300)) + ((var63 - 3) * 15)) + ((var60 + (var64 * 2)) - 4)];
                var7 = (((var4 * 6) + (var5 * 6)) + (var62 * 2)) + var64;
                var2.data[var6] = var2.data[var6] + (var66 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var67 = 6; var67 < 7; var67 = var67 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var42 * var2.width)) + var67;
            int var68 = var67 * 3;
            int var69 = var42 * 2;
            for (int var70 = 0; var70 < 3; var70 = var70 + 1) {
              int var71 = var69 + (var70 * 3);
              for (int var72 = 0; var72 < 2; var72 = var72 + 1) {
                int var73 = var68 + (var72 * 2);
                if (var73 < 4) {
                  continue;
                } 
                if (var73 < 19) {
                  int var74 = arg0[(((var3 * 300) + (var5 * 300)) + ((var71 - 3) * 15)) + (var73 - 4)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var70 * 2)) + var72;
                  var2.data[var6] = var2.data[var6] + (var74 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var75 = 9; var75 < 10; var75 = var75 + 1) {
          for (int var76 = 0; var76 < 1; var76 = var76 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var75 * var2.width)) + var76;
            int var77 = var76 * 3;
            int var78 = var75 * 2;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = var78 + (var79 * 3);
              if (var80 < 3) {
                continue;
              } 
              if (var80 < 23) {
                for (int var81 = 0; var81 < 2; var81 = var81 + 1) {
                  int var82 = var77 + (var81 * 2);
                  if (var82 < 4) {
                    continue;
                  } 
                  if (var82 < 19) {
                    int var83 = arg0[(((var3 * 300) + (var5 * 300)) + ((var80 - 3) * 15)) + (var82 - 4)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var79 * 2)) + var81;
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
          for (int var84 = 1; var84 < 2; var84 = var84 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var75 * var2.width)) + var84;
            int var85 = var84 * 3;
            int var86 = var75 * 2;
            for (int var87 = 0; var87 < 3; var87 = var87 + 1) {
              int var88 = var86 + (var87 * 3);
              if (var88 < 3) {
                continue;
              } 
              if (var88 < 23) {
                for (int var89 = 0; var89 < 2; var89 = var89 + 1) {
                  int var90 = var85 + (var89 * 2);
                  if (var90 < 4) {
                    continue;
                  } 
                  if (var90 < 19) {
                    int var91 = arg0[(((var3 * 300) + (var5 * 300)) + ((var88 - 3) * 15)) + (var90 - 4)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var87 * 2)) + var89;
                    var2.data[var6] = var2.data[var6] + (var91 * arg1[var7]);
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
          for (int var92 = 2; var92 < 6; var92 = var92 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var75 * var2.width)) + var92;
            int var93 = var92 * 3;
            int var94 = var75 * 2;
            for (int var95 = 0; var95 < 3; var95 = var95 + 1) {
              int var96 = var94 + (var95 * 3);
              if (var96 < 3) {
                continue;
              } 
              if (var96 < 23) {
                for (int var97 = 0; var97 < 2; var97 = var97 + 1) {
                  int var99 = arg0[(((var3 * 300) + (var5 * 300)) + ((var96 - 3) * 15)) + ((var93 + (var97 * 2)) - 4)];
                  var7 = (((var4 * 6) + (var5 * 6)) + (var95 * 2)) + var97;
                  var2.data[var6] = var2.data[var6] + (var99 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var100 = 6; var100 < 7; var100 = var100 + 1) {
            var6 = (((var3 * 70) + (var4 * 70)) + (var75 * var2.width)) + var100;
            int var101 = var100 * 3;
            int var102 = var75 * 2;
            for (int var103 = 0; var103 < 3; var103 = var103 + 1) {
              int var104 = var102 + (var103 * 3);
              if (var104 < 3) {
                continue;
              } 
              if (var104 < 23) {
                for (int var105 = 0; var105 < 2; var105 = var105 + 1) {
                  int var106 = var101 + (var105 * 2);
                  if (var106 < 4) {
                    continue;
                  } 
                  if (var106 < 19) {
                    int var107 = arg0[(((var3 * 300) + (var5 * 300)) + ((var104 - 3) * 15)) + (var106 - 4)];
                    var7 = (((var4 * 6) + (var5 * 6)) + (var103 * 2)) + var105;
                    var2.data[var6] = var2.data[var6] + (var107 * arg1[var7]);
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
            var6 = (((var3 * 300) + (var4 * 300)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 1;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 3);
              if (var14 < 3) {
                continue;
              } 
              if (var14 < 23) {
                for (int var15 = 0; var15 < 3; var15 = var15 + 1) {
                  int var16 = var11 + (var15 * 2);
                  if (var16 < 2) {
                    continue;
                  } 
                  if (var16 < 17) {
                    int var17 = arg0[(((var3 * 300) + (var5 * 300)) + ((var14 - 3) * 15)) + (var16 - 2)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var13 * 3)) + var15;
                    var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
          for (int var18 = 2; var18 < 13; var18 = var18 + 1) {
            var6 = (((var3 * 300) + (var4 * 300)) + (var9 * var2.width)) + var18;
            int var19 = var18 * 1;
            int var20 = var9 * 1;
            for (int var21 = 0; var21 < 3; var21 = var21 + 1) {
              int var22 = var20 + (var21 * 3);
              if (var22 < 3) {
                continue;
              } 
              if (var22 < 23) {
                for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
                  int var25 = arg0[(((var3 * 300) + (var5 * 300)) + ((var22 - 3) * 15)) + ((var19 + (var23 * 2)) - 2)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var21 * 3)) + var23;
                  var2.data[var6] = var2.data[var6] + (var25 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var26 = 13; var26 < 15; var26 = var26 + 1) {
            var6 = (((var3 * 300) + (var4 * 300)) + (var9 * var2.width)) + var26;
            int var27 = var26 * 1;
            int var28 = var9 * 1;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = var28 + (var29 * 3);
              if (var30 < 3) {
                continue;
              } 
              if (var30 < 23) {
                for (int var31 = 0; var31 < 3; var31 = var31 + 1) {
                  int var32 = var27 + (var31 * 2);
                  if (var32 < 2) {
                    continue;
                  } 
                  if (var32 < 17) {
                    int var33 = arg0[(((var3 * 300) + (var5 * 300)) + ((var30 - 3) * 15)) + (var32 - 2)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var29 * 3)) + var31;
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
        for (int var34 = 3; var34 < 17; var34 = var34 + 1) {
          for (int var35 = 0; var35 < 2; var35 = var35 + 1) {
            var6 = (((var3 * 300) + (var4 * 300)) + (var34 * var2.width)) + var35;
            int var36 = var35 * 1;
            int var37 = var34 * 1;
            for (int var38 = 0; var38 < 3; var38 = var38 + 1) {
              int var39 = var37 + (var38 * 3);
              for (int var40 = 0; var40 < 3; var40 = var40 + 1) {
                int var41 = var36 + (var40 * 2);
                if (var41 < 2) {
                  continue;
                } 
                if (var41 < 17) {
                  int var42 = arg0[(((var3 * 300) + (var5 * 300)) + ((var39 - 3) * 15)) + (var41 - 2)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var38 * 3)) + var40;
                  var2.data[var6] = var2.data[var6] + (var42 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var43 = 2; var43 < 13; var43 = var43 + 1) {
            var6 = (((var3 * 300) + (var4 * 300)) + (var34 * var2.width)) + var43;
            int var44 = var43 * 1;
            int var45 = var34 * 1;
            for (int var46 = 0; var46 < 3; var46 = var46 + 1) {
              int var47 = var45 + (var46 * 3);
              for (int var48 = 0; var48 < 3; var48 = var48 + 1) {
                int var50 = arg0[(((var3 * 300) + (var5 * 300)) + ((var47 - 3) * 15)) + ((var44 + (var48 * 2)) - 2)];
                var7 = (((var4 * 9) + (var5 * 9)) + (var46 * 3)) + var48;
                var2.data[var6] = var2.data[var6] + (var50 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var51 = 13; var51 < 15; var51 = var51 + 1) {
            var6 = (((var3 * 300) + (var4 * 300)) + (var34 * var2.width)) + var51;
            int var52 = var51 * 1;
            int var53 = var34 * 1;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = var53 + (var54 * 3);
              for (int var56 = 0; var56 < 3; var56 = var56 + 1) {
                int var57 = var52 + (var56 * 2);
                if (var57 < 2) {
                  continue;
                } 
                if (var57 < 17) {
                  int var58 = arg0[(((var3 * 300) + (var5 * 300)) + ((var55 - 3) * 15)) + (var57 - 2)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var54 * 3)) + var56;
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
        for (int var59 = 17; var59 < 20; var59 = var59 + 1) {
          for (int var60 = 0; var60 < 2; var60 = var60 + 1) {
            var6 = (((var3 * 300) + (var4 * 300)) + (var59 * var2.width)) + var60;
            int var61 = var60 * 1;
            int var62 = var59 * 1;
            for (int var63 = 0; var63 < 3; var63 = var63 + 1) {
              int var64 = var62 + (var63 * 3);
              if (var64 < 3) {
                continue;
              } 
              if (var64 < 23) {
                for (int var65 = 0; var65 < 3; var65 = var65 + 1) {
                  int var66 = var61 + (var65 * 2);
                  if (var66 < 2) {
                    continue;
                  } 
                  if (var66 < 17) {
                    int var67 = arg0[(((var3 * 300) + (var5 * 300)) + ((var64 - 3) * 15)) + (var66 - 2)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var63 * 3)) + var65;
                    var2.data[var6] = var2.data[var6] + (var67 * arg1[var7]);
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
          for (int var68 = 2; var68 < 13; var68 = var68 + 1) {
            var6 = (((var3 * 300) + (var4 * 300)) + (var59 * var2.width)) + var68;
            int var69 = var68 * 1;
            int var70 = var59 * 1;
            for (int var71 = 0; var71 < 3; var71 = var71 + 1) {
              int var72 = var70 + (var71 * 3);
              if (var72 < 3) {
                continue;
              } 
              if (var72 < 23) {
                for (int var73 = 0; var73 < 3; var73 = var73 + 1) {
                  int var75 = arg0[(((var3 * 300) + (var5 * 300)) + ((var72 - 3) * 15)) + ((var69 + (var73 * 2)) - 2)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var71 * 3)) + var73;
                  var2.data[var6] = var2.data[var6] + (var75 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var76 = 13; var76 < 15; var76 = var76 + 1) {
            var6 = (((var3 * 300) + (var4 * 300)) + (var59 * var2.width)) + var76;
            int var77 = var76 * 1;
            int var78 = var59 * 1;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = var78 + (var79 * 3);
              if (var80 < 3) {
                continue;
              } 
              if (var80 < 23) {
                for (int var81 = 0; var81 < 3; var81 = var81 + 1) {
                  int var82 = var77 + (var81 * 2);
                  if (var82 < 2) {
                    continue;
                  } 
                  if (var82 < 17) {
                    int var83 = arg0[(((var3 * 300) + (var5 * 300)) + ((var80 - 3) * 15)) + (var82 - 2)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var79 * 3)) + var81;
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
            var6 = (((var3 * 78) + (var4 * 78)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 4;
            int var12 = var9 * 2;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 2);
              if (var14 < 5) {
                continue;
              } 
              if (var14 < 25) {
                for (int var15 = 0; var15 < 3; var15 = var15 + 1) {
                  int var16 = var11 + (var15 * 2);
                  if (var16 < 4) {
                    continue;
                  } 
                  if (var16 < 24) {
                    int var17 = arg0[(((var3 * 400) + (var5 * 400)) + ((var14 - 5) * 20)) + (var16 - 4)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var13 * 3)) + var15;
                    var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
          for (int var18 = 1; var18 < 5; var18 = var18 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var9 * var2.width)) + var18;
            int var19 = var18 * 4;
            int var20 = var9 * 2;
            for (int var21 = 0; var21 < 3; var21 = var21 + 1) {
              int var22 = var20 + (var21 * 2);
              if (var22 < 5) {
                continue;
              } 
              if (var22 < 25) {
                for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
                  int var25 = arg0[(((var3 * 400) + (var5 * 400)) + ((var22 - 5) * 20)) + ((var19 + (var23 * 2)) - 4)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var21 * 3)) + var23;
                  var2.data[var6] = var2.data[var6] + (var25 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var26 = 5; var26 < 6; var26 = var26 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var9 * var2.width)) + var26;
            int var27 = var26 * 4;
            int var28 = var9 * 2;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = var28 + (var29 * 2);
              if (var30 < 5) {
                continue;
              } 
              if (var30 < 25) {
                for (int var31 = 0; var31 < 3; var31 = var31 + 1) {
                  int var32 = var27 + (var31 * 2);
                  if (var32 < 4) {
                    continue;
                  } 
                  if (var32 < 24) {
                    int var33 = arg0[(((var3 * 400) + (var5 * 400)) + ((var30 - 5) * 20)) + (var32 - 4)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var29 * 3)) + var31;
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
        for (int var34 = 1; var34 < 3; var34 = var34 + 1) {
          for (int var35 = 0; var35 < 1; var35 = var35 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var34 * var2.width)) + var35;
            int var36 = var35 * 4;
            int var37 = var34 * 2;
            for (int var38 = 0; var38 < 3; var38 = var38 + 1) {
              int var39 = var37 + (var38 * 2);
              if (var39 < 5) {
                continue;
              } 
              if (var39 < 25) {
                for (int var40 = 0; var40 < 3; var40 = var40 + 1) {
                  int var41 = var36 + (var40 * 2);
                  if (var41 < 4) {
                    continue;
                  } 
                  if (var41 < 24) {
                    int var42 = arg0[(((var3 * 400) + (var5 * 400)) + ((var39 - 5) * 20)) + (var41 - 4)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var38 * 3)) + var40;
                    var2.data[var6] = var2.data[var6] + (var42 * arg1[var7]);
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
          for (int var43 = 1; var43 < 5; var43 = var43 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var34 * var2.width)) + var43;
            int var44 = var43 * 4;
            int var45 = var34 * 2;
            for (int var46 = 0; var46 < 3; var46 = var46 + 1) {
              int var47 = var45 + (var46 * 2);
              if (var47 < 5) {
                continue;
              } 
              if (var47 < 25) {
                for (int var48 = 0; var48 < 3; var48 = var48 + 1) {
                  int var50 = arg0[(((var3 * 400) + (var5 * 400)) + ((var47 - 5) * 20)) + ((var44 + (var48 * 2)) - 4)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var46 * 3)) + var48;
                  var2.data[var6] = var2.data[var6] + (var50 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var51 = 5; var51 < 6; var51 = var51 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var34 * var2.width)) + var51;
            int var52 = var51 * 4;
            int var53 = var34 * 2;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = var53 + (var54 * 2);
              if (var55 < 5) {
                continue;
              } 
              if (var55 < 25) {
                for (int var56 = 0; var56 < 3; var56 = var56 + 1) {
                  int var57 = var52 + (var56 * 2);
                  if (var57 < 4) {
                    continue;
                  } 
                  if (var57 < 24) {
                    int var58 = arg0[(((var3 * 400) + (var5 * 400)) + ((var55 - 5) * 20)) + (var57 - 4)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var54 * 3)) + var56;
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
        for (int var59 = 3; var59 < 11; var59 = var59 + 1) {
          for (int var60 = 0; var60 < 1; var60 = var60 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var59 * var2.width)) + var60;
            int var61 = var60 * 4;
            int var62 = var59 * 2;
            for (int var63 = 0; var63 < 3; var63 = var63 + 1) {
              int var64 = var62 + (var63 * 2);
              for (int var65 = 0; var65 < 3; var65 = var65 + 1) {
                int var66 = var61 + (var65 * 2);
                if (var66 < 4) {
                  continue;
                } 
                if (var66 < 24) {
                  int var67 = arg0[(((var3 * 400) + (var5 * 400)) + ((var64 - 5) * 20)) + (var66 - 4)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var63 * 3)) + var65;
                  var2.data[var6] = var2.data[var6] + (var67 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var68 = 1; var68 < 5; var68 = var68 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var59 * var2.width)) + var68;
            int var69 = var68 * 4;
            int var70 = var59 * 2;
            for (int var71 = 0; var71 < 3; var71 = var71 + 1) {
              int var72 = var70 + (var71 * 2);
              for (int var73 = 0; var73 < 3; var73 = var73 + 1) {
                int var75 = arg0[(((var3 * 400) + (var5 * 400)) + ((var72 - 5) * 20)) + ((var69 + (var73 * 2)) - 4)];
                var7 = (((var4 * 9) + (var5 * 9)) + (var71 * 3)) + var73;
                var2.data[var6] = var2.data[var6] + (var75 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var76 = 5; var76 < 6; var76 = var76 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var59 * var2.width)) + var76;
            int var77 = var76 * 4;
            int var78 = var59 * 2;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = var78 + (var79 * 2);
              for (int var81 = 0; var81 < 3; var81 = var81 + 1) {
                int var82 = var77 + (var81 * 2);
                if (var82 < 4) {
                  continue;
                } 
                if (var82 < 24) {
                  int var83 = arg0[(((var3 * 400) + (var5 * 400)) + ((var80 - 5) * 20)) + (var82 - 4)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var79 * 3)) + var81;
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
        for (int var84 = 11; var84 < 13; var84 = var84 + 1) {
          for (int var85 = 0; var85 < 1; var85 = var85 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var84 * var2.width)) + var85;
            int var86 = var85 * 4;
            int var87 = var84 * 2;
            for (int var88 = 0; var88 < 3; var88 = var88 + 1) {
              int var89 = var87 + (var88 * 2);
              if (var89 < 5) {
                continue;
              } 
              if (var89 < 25) {
                for (int var90 = 0; var90 < 3; var90 = var90 + 1) {
                  int var91 = var86 + (var90 * 2);
                  if (var91 < 4) {
                    continue;
                  } 
                  if (var91 < 24) {
                    int var92 = arg0[(((var3 * 400) + (var5 * 400)) + ((var89 - 5) * 20)) + (var91 - 4)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var88 * 3)) + var90;
                    var2.data[var6] = var2.data[var6] + (var92 * arg1[var7]);
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
          for (int var93 = 1; var93 < 5; var93 = var93 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var84 * var2.width)) + var93;
            int var94 = var93 * 4;
            int var95 = var84 * 2;
            for (int var96 = 0; var96 < 3; var96 = var96 + 1) {
              int var97 = var95 + (var96 * 2);
              if (var97 < 5) {
                continue;
              } 
              if (var97 < 25) {
                for (int var98 = 0; var98 < 3; var98 = var98 + 1) {
                  int var100 = arg0[(((var3 * 400) + (var5 * 400)) + ((var97 - 5) * 20)) + ((var94 + (var98 * 2)) - 4)];
                  var7 = (((var4 * 9) + (var5 * 9)) + (var96 * 3)) + var98;
                  var2.data[var6] = var2.data[var6] + (var100 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var101 = 5; var101 < 6; var101 = var101 + 1) {
            var6 = (((var3 * 78) + (var4 * 78)) + (var84 * var2.width)) + var101;
            int var102 = var101 * 4;
            int var103 = var84 * 2;
            for (int var104 = 0; var104 < 3; var104 = var104 + 1) {
              int var105 = var103 + (var104 * 2);
              if (var105 < 5) {
                continue;
              } 
              if (var105 < 25) {
                for (int var106 = 0; var106 < 3; var106 = var106 + 1) {
                  int var107 = var102 + (var106 * 2);
                  if (var107 < 4) {
                    continue;
                  } 
                  if (var107 < 24) {
                    int var108 = arg0[(((var3 * 400) + (var5 * 400)) + ((var105 - 5) * 20)) + (var107 - 4)];
                    var7 = (((var4 * 9) + (var5 * 9)) + (var104 * 3)) + var106;
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
            var6 = (((var3 * 275) + (var4 * 55)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 4;
            int var12 = var9 * 2;
            for (int var13 = 0; var13 < 5; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 2);
              if (var14 < 5) {
                continue;
              } 
              if (var14 < 25) {
                for (int var15 = 0; var15 < 5; var15 = var15 + 1) {
                  int var16 = var11 + (var15 * 2);
                  if (var16 < 4) {
                    continue;
                  } 
                  if (var16 < 24) {
                    int var17 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var14 - 5) * 20)) + (var16 - 4)];
                    var7 = (((var4 * 75) + (var5 * 25)) + (var13 * 5)) + var15;
                    var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
          for (int var18 = 1; var18 < 4; var18 = var18 + 1) {
            var6 = (((var3 * 275) + (var4 * 55)) + (var9 * var2.width)) + var18;
            int var19 = var18 * 4;
            int var20 = var9 * 2;
            for (int var21 = 0; var21 < 5; var21 = var21 + 1) {
              int var22 = var20 + (var21 * 2);
              if (var22 < 5) {
                continue;
              } 
              if (var22 < 25) {
                for (int var23 = 0; var23 < 5; var23 = var23 + 1) {
                  int var25 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var22 - 5) * 20)) + ((var19 + (var23 * 2)) - 4)];
                  var7 = (((var4 * 75) + (var5 * 25)) + (var21 * 5)) + var23;
                  var2.data[var6] = var2.data[var6] + (var25 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var26 = 4; var26 < 5; var26 = var26 + 1) {
            var6 = (((var3 * 275) + (var4 * 55)) + (var9 * var2.width)) + var26;
            int var27 = var26 * 4;
            int var28 = var9 * 2;
            for (int var29 = 0; var29 < 5; var29 = var29 + 1) {
              int var30 = var28 + (var29 * 2);
              if (var30 < 5) {
                continue;
              } 
              if (var30 < 25) {
                for (int var31 = 0; var31 < 5; var31 = var31 + 1) {
                  int var32 = var27 + (var31 * 2);
                  if (var32 < 4) {
                    continue;
                  } 
                  if (var32 < 24) {
                    int var33 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var30 - 5) * 20)) + (var32 - 4)];
                    var7 = (((var4 * 75) + (var5 * 25)) + (var29 * 5)) + var31;
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
        for (int var34 = 3; var34 < 9; var34 = var34 + 1) {
          for (int var35 = 0; var35 < 1; var35 = var35 + 1) {
            var6 = (((var3 * 275) + (var4 * 55)) + (var34 * var2.width)) + var35;
            int var36 = var35 * 4;
            int var37 = var34 * 2;
            for (int var38 = 0; var38 < 5; var38 = var38 + 1) {
              int var39 = var37 + (var38 * 2);
              for (int var40 = 0; var40 < 5; var40 = var40 + 1) {
                int var41 = var36 + (var40 * 2);
                if (var41 < 4) {
                  continue;
                } 
                if (var41 < 24) {
                  int var42 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var39 - 5) * 20)) + (var41 - 4)];
                  var7 = (((var4 * 75) + (var5 * 25)) + (var38 * 5)) + var40;
                  var2.data[var6] = var2.data[var6] + (var42 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var43 = 1; var43 < 4; var43 = var43 + 1) {
            var6 = (((var3 * 275) + (var4 * 55)) + (var34 * var2.width)) + var43;
            int var44 = var43 * 4;
            int var45 = var34 * 2;
            for (int var46 = 0; var46 < 5; var46 = var46 + 1) {
              int var47 = var45 + (var46 * 2);
              for (int var48 = 0; var48 < 5; var48 = var48 + 1) {
                int var50 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var47 - 5) * 20)) + ((var44 + (var48 * 2)) - 4)];
                var7 = (((var4 * 75) + (var5 * 25)) + (var46 * 5)) + var48;
                var2.data[var6] = var2.data[var6] + (var50 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var51 = 4; var51 < 5; var51 = var51 + 1) {
            var6 = (((var3 * 275) + (var4 * 55)) + (var34 * var2.width)) + var51;
            int var52 = var51 * 4;
            int var53 = var34 * 2;
            for (int var54 = 0; var54 < 5; var54 = var54 + 1) {
              int var55 = var53 + (var54 * 2);
              for (int var56 = 0; var56 < 5; var56 = var56 + 1) {
                int var57 = var52 + (var56 * 2);
                if (var57 < 4) {
                  continue;
                } 
                if (var57 < 24) {
                  int var58 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var55 - 5) * 20)) + (var57 - 4)];
                  var7 = (((var4 * 75) + (var5 * 25)) + (var54 * 5)) + var56;
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
        for (int var59 = 9; var59 < 11; var59 = var59 + 1) {
          for (int var60 = 0; var60 < 1; var60 = var60 + 1) {
            var6 = (((var3 * 275) + (var4 * 55)) + (var59 * var2.width)) + var60;
            int var61 = var60 * 4;
            int var62 = var59 * 2;
            for (int var63 = 0; var63 < 5; var63 = var63 + 1) {
              int var64 = var62 + (var63 * 2);
              if (var64 < 5) {
                continue;
              } 
              if (var64 < 25) {
                for (int var65 = 0; var65 < 5; var65 = var65 + 1) {
                  int var66 = var61 + (var65 * 2);
                  if (var66 < 4) {
                    continue;
                  } 
                  if (var66 < 24) {
                    int var67 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var64 - 5) * 20)) + (var66 - 4)];
                    var7 = (((var4 * 75) + (var5 * 25)) + (var63 * 5)) + var65;
                    var2.data[var6] = var2.data[var6] + (var67 * arg1[var7]);
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
          for (int var68 = 1; var68 < 4; var68 = var68 + 1) {
            var6 = (((var3 * 275) + (var4 * 55)) + (var59 * var2.width)) + var68;
            int var69 = var68 * 4;
            int var70 = var59 * 2;
            for (int var71 = 0; var71 < 5; var71 = var71 + 1) {
              int var72 = var70 + (var71 * 2);
              if (var72 < 5) {
                continue;
              } 
              if (var72 < 25) {
                for (int var73 = 0; var73 < 5; var73 = var73 + 1) {
                  int var75 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var72 - 5) * 20)) + ((var69 + (var73 * 2)) - 4)];
                  var7 = (((var4 * 75) + (var5 * 25)) + (var71 * 5)) + var73;
                  var2.data[var6] = var2.data[var6] + (var75 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var76 = 4; var76 < 5; var76 = var76 + 1) {
            var6 = (((var3 * 275) + (var4 * 55)) + (var59 * var2.width)) + var76;
            int var77 = var76 * 4;
            int var78 = var59 * 2;
            for (int var79 = 0; var79 < 5; var79 = var79 + 1) {
              int var80 = var78 + (var79 * 2);
              if (var80 < 5) {
                continue;
              } 
              if (var80 < 25) {
                for (int var81 = 0; var81 < 5; var81 = var81 + 1) {
                  int var82 = var77 + (var81 * 2);
                  if (var82 < 4) {
                    continue;
                  } 
                  if (var82 < 24) {
                    int var83 = arg0[(((var3 * 1200) + (var5 * 400)) + ((var80 - 5) * 20)) + (var82 - 4)];
                    var7 = (((var4 * 75) + (var5 * 25)) + (var79 * 5)) + var81;
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
            var6 = (((var3 * 82810) + (var4 * 8281)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 1;
            for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 1);
              for (int var15 = 0; var15 < 10; var15 = var15 + 1) {
                int var17 = arg0[(((var3 * 100000) + (var5 * 10000)) + ((var14 - 0) * 100)) + ((var11 + (var15 * 1)) - 0)];
                var7 = (((var4 * 1000) + (var5 * 100)) + (var13 * 10)) + var15;
                var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
            var6 = (((var3 * 5290) + (var4 * 529)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 4;
            int var12 = var9 * 4;
            for (int var13 = 0; var13 < 10; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 1);
              for (int var15 = 0; var15 < 10; var15 = var15 + 1) {
                int var17 = arg0[(((var3 * 50000) + (var5 * 10000)) + ((var14 - 0) * 100)) + ((var11 + (var15 * 1)) - 0)];
                var7 = (((var4 * 500) + (var5 * 100)) + (var13 * 10)) + var15;
                var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
            var6 = (((var3 * 36) + (var4 * 36)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 1;
            for (int var13 = 0; var13 < 5; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 1);
              for (int var15 = 0; var15 < 5; var15 = var15 + 1) {
                int var17 = arg0[(((var3 * 500) + (var5 * 100)) + ((var14 - 0) * 10)) + ((var11 + (var15 * 1)) - 0)];
                var7 = (((var4 * 125) + (var5 * 25)) + (var13 * 5)) + var15;
                var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
            var6 = (((var3 * 1452) + (var4 * 484)) + (var9 * var2.width)) + var10;
            int var11 = var10 * 1;
            int var12 = var9 * 1;
            for (int var13 = 0; var13 < 3; var13 = var13 + 1) {
              int var14 = var12 + (var13 * 4);
              if (var14 < 10) {
                continue;
              } 
              if (var14 < 20) {
                for (int var15 = 0; var15 < 3; var15 = var15 + 1) {
                  int var16 = var11 + (var15 * 4);
                  if (var16 < 10) {
                    continue;
                  } 
                  if (var16 < 20) {
                    int var17 = arg0[(((var3 * 500) + (var5 * 100)) + ((var14 - 10) * 10)) + (var16 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var13 * 3)) + var15;
                    var2.data[var6] = var2.data[var6] + (var17 * arg1[var7]);
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
          for (int var18 = 2; var18 < 10; var18 = var18 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var9 * var2.width)) + var18;
            int var19 = var18 * 1;
            int var20 = var9 * 1;
            for (int var21 = 0; var21 < 3; var21 = var21 + 1) {
              int var22 = var20 + (var21 * 4);
              if (var22 < 10) {
                continue;
              } 
              if (var22 < 20) {
                for (int var23 = 0; var23 < 3; var23 = var23 + 1) {
                  int var24 = var19 + (var23 * 4);
                  if (var24 < 10) {
                    continue;
                  } 
                  if (var24 < 20) {
                    int var25 = arg0[(((var3 * 500) + (var5 * 100)) + ((var22 - 10) * 10)) + (var24 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var21 * 3)) + var23;
                    var2.data[var6] = var2.data[var6] + (var25 * arg1[var7]);
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
          for (int var26 = 10; var26 < 12; var26 = var26 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var9 * var2.width)) + var26;
            int var27 = var26 * 1;
            int var28 = var9 * 1;
            for (int var29 = 0; var29 < 3; var29 = var29 + 1) {
              int var30 = var28 + (var29 * 4);
              if (var30 < 10) {
                continue;
              } 
              if (var30 < 20) {
                for (int var31 = 0; var31 < 3; var31 = var31 + 1) {
                  int var33 = arg0[(((var3 * 500) + (var5 * 100)) + ((var30 - 10) * 10)) + ((var27 + (var31 * 4)) - 10)];
                  var7 = (((var4 * 45) + (var5 * 9)) + (var29 * 3)) + var31;
                  var2.data[var6] = var2.data[var6] + (var33 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var34 = 12; var34 < 22; var34 = var34 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var9 * var2.width)) + var34;
            int var35 = var34 * 1;
            int var36 = var9 * 1;
            for (int var37 = 0; var37 < 3; var37 = var37 + 1) {
              int var38 = var36 + (var37 * 4);
              if (var38 < 10) {
                continue;
              } 
              if (var38 < 20) {
                for (int var39 = 0; var39 < 3; var39 = var39 + 1) {
                  int var40 = var35 + (var39 * 4);
                  if (var40 < 10) {
                    continue;
                  } 
                  if (var40 < 20) {
                    int var41 = arg0[(((var3 * 500) + (var5 * 100)) + ((var38 - 10) * 10)) + (var40 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var37 * 3)) + var39;
                    var2.data[var6] = var2.data[var6] + (var41 * arg1[var7]);
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
        for (int var42 = 2; var42 < 10; var42 = var42 + 1) {
          for (int var43 = 0; var43 < 2; var43 = var43 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var42 * var2.width)) + var43;
            int var44 = var43 * 1;
            int var45 = var42 * 1;
            for (int var46 = 0; var46 < 3; var46 = var46 + 1) {
              int var47 = var45 + (var46 * 4);
              if (var47 < 10) {
                continue;
              } 
              if (var47 < 20) {
                for (int var48 = 0; var48 < 3; var48 = var48 + 1) {
                  int var49 = var44 + (var48 * 4);
                  if (var49 < 10) {
                    continue;
                  } 
                  if (var49 < 20) {
                    int var50 = arg0[(((var3 * 500) + (var5 * 100)) + ((var47 - 10) * 10)) + (var49 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var46 * 3)) + var48;
                    var2.data[var6] = var2.data[var6] + (var50 * arg1[var7]);
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
          for (int var51 = 2; var51 < 10; var51 = var51 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var42 * var2.width)) + var51;
            int var52 = var51 * 1;
            int var53 = var42 * 1;
            for (int var54 = 0; var54 < 3; var54 = var54 + 1) {
              int var55 = var53 + (var54 * 4);
              if (var55 < 10) {
                continue;
              } 
              if (var55 < 20) {
                for (int var56 = 0; var56 < 3; var56 = var56 + 1) {
                  int var57 = var52 + (var56 * 4);
                  if (var57 < 10) {
                    continue;
                  } 
                  if (var57 < 20) {
                    int var58 = arg0[(((var3 * 500) + (var5 * 100)) + ((var55 - 10) * 10)) + (var57 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var54 * 3)) + var56;
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
          for (int var59 = 10; var59 < 12; var59 = var59 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var42 * var2.width)) + var59;
            int var60 = var59 * 1;
            int var61 = var42 * 1;
            for (int var62 = 0; var62 < 3; var62 = var62 + 1) {
              int var63 = var61 + (var62 * 4);
              if (var63 < 10) {
                continue;
              } 
              if (var63 < 20) {
                for (int var64 = 0; var64 < 3; var64 = var64 + 1) {
                  int var66 = arg0[(((var3 * 500) + (var5 * 100)) + ((var63 - 10) * 10)) + ((var60 + (var64 * 4)) - 10)];
                  var7 = (((var4 * 45) + (var5 * 9)) + (var62 * 3)) + var64;
                  var2.data[var6] = var2.data[var6] + (var66 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var67 = 12; var67 < 22; var67 = var67 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var42 * var2.width)) + var67;
            int var68 = var67 * 1;
            int var69 = var42 * 1;
            for (int var70 = 0; var70 < 3; var70 = var70 + 1) {
              int var71 = var69 + (var70 * 4);
              if (var71 < 10) {
                continue;
              } 
              if (var71 < 20) {
                for (int var72 = 0; var72 < 3; var72 = var72 + 1) {
                  int var73 = var68 + (var72 * 4);
                  if (var73 < 10) {
                    continue;
                  } 
                  if (var73 < 20) {
                    int var74 = arg0[(((var3 * 500) + (var5 * 100)) + ((var71 - 10) * 10)) + (var73 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var70 * 3)) + var72;
                    var2.data[var6] = var2.data[var6] + (var74 * arg1[var7]);
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
        for (int var75 = 10; var75 < 12; var75 = var75 + 1) {
          for (int var76 = 0; var76 < 2; var76 = var76 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var75 * var2.width)) + var76;
            int var77 = var76 * 1;
            int var78 = var75 * 1;
            for (int var79 = 0; var79 < 3; var79 = var79 + 1) {
              int var80 = var78 + (var79 * 4);
              for (int var81 = 0; var81 < 3; var81 = var81 + 1) {
                int var82 = var77 + (var81 * 4);
                if (var82 < 10) {
                  continue;
                } 
                if (var82 < 20) {
                  int var83 = arg0[(((var3 * 500) + (var5 * 100)) + ((var80 - 10) * 10)) + (var82 - 10)];
                  var7 = (((var4 * 45) + (var5 * 9)) + (var79 * 3)) + var81;
                  var2.data[var6] = var2.data[var6] + (var83 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var84 = 2; var84 < 10; var84 = var84 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var75 * var2.width)) + var84;
            int var85 = var84 * 1;
            int var86 = var75 * 1;
            for (int var87 = 0; var87 < 3; var87 = var87 + 1) {
              int var88 = var86 + (var87 * 4);
              for (int var89 = 0; var89 < 3; var89 = var89 + 1) {
                int var90 = var85 + (var89 * 4);
                if (var90 < 10) {
                  continue;
                } 
                if (var90 < 20) {
                  int var91 = arg0[(((var3 * 500) + (var5 * 100)) + ((var88 - 10) * 10)) + (var90 - 10)];
                  var7 = (((var4 * 45) + (var5 * 9)) + (var87 * 3)) + var89;
                  var2.data[var6] = var2.data[var6] + (var91 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
          for (int var92 = 10; var92 < 12; var92 = var92 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var75 * var2.width)) + var92;
            int var93 = var92 * 1;
            int var94 = var75 * 1;
            for (int var95 = 0; var95 < 3; var95 = var95 + 1) {
              int var96 = var94 + (var95 * 4);
              for (int var97 = 0; var97 < 3; var97 = var97 + 1) {
                int var99 = arg0[(((var3 * 500) + (var5 * 100)) + ((var96 - 10) * 10)) + ((var93 + (var97 * 4)) - 10)];
                var7 = (((var4 * 45) + (var5 * 9)) + (var95 * 3)) + var97;
                var2.data[var6] = var2.data[var6] + (var99 * arg1[var7]);
                var8 = var8 + 1;
              }
            }
          }
          for (int var100 = 12; var100 < 22; var100 = var100 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var75 * var2.width)) + var100;
            int var101 = var100 * 1;
            int var102 = var75 * 1;
            for (int var103 = 0; var103 < 3; var103 = var103 + 1) {
              int var104 = var102 + (var103 * 4);
              for (int var105 = 0; var105 < 3; var105 = var105 + 1) {
                int var106 = var101 + (var105 * 4);
                if (var106 < 10) {
                  continue;
                } 
                if (var106 < 20) {
                  int var107 = arg0[(((var3 * 500) + (var5 * 100)) + ((var104 - 10) * 10)) + (var106 - 10)];
                  var7 = (((var4 * 45) + (var5 * 9)) + (var103 * 3)) + var105;
                  var2.data[var6] = var2.data[var6] + (var107 * arg1[var7]);
                  var8 = var8 + 1;
                } else {
                  break;
                }
              }
            }
          }
        }
        // looping over the output
        for (int var108 = 12; var108 < 22; var108 = var108 + 1) {
          for (int var109 = 0; var109 < 2; var109 = var109 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var108 * var2.width)) + var109;
            int var110 = var109 * 1;
            int var111 = var108 * 1;
            for (int var112 = 0; var112 < 3; var112 = var112 + 1) {
              int var113 = var111 + (var112 * 4);
              if (var113 < 10) {
                continue;
              } 
              if (var113 < 20) {
                for (int var114 = 0; var114 < 3; var114 = var114 + 1) {
                  int var115 = var110 + (var114 * 4);
                  if (var115 < 10) {
                    continue;
                  } 
                  if (var115 < 20) {
                    int var116 = arg0[(((var3 * 500) + (var5 * 100)) + ((var113 - 10) * 10)) + (var115 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var112 * 3)) + var114;
                    var2.data[var6] = var2.data[var6] + (var116 * arg1[var7]);
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
          for (int var117 = 2; var117 < 10; var117 = var117 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var108 * var2.width)) + var117;
            int var118 = var117 * 1;
            int var119 = var108 * 1;
            for (int var120 = 0; var120 < 3; var120 = var120 + 1) {
              int var121 = var119 + (var120 * 4);
              if (var121 < 10) {
                continue;
              } 
              if (var121 < 20) {
                for (int var122 = 0; var122 < 3; var122 = var122 + 1) {
                  int var123 = var118 + (var122 * 4);
                  if (var123 < 10) {
                    continue;
                  } 
                  if (var123 < 20) {
                    int var124 = arg0[(((var3 * 500) + (var5 * 100)) + ((var121 - 10) * 10)) + (var123 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var120 * 3)) + var122;
                    var2.data[var6] = var2.data[var6] + (var124 * arg1[var7]);
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
          for (int var125 = 10; var125 < 12; var125 = var125 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var108 * var2.width)) + var125;
            int var126 = var125 * 1;
            int var127 = var108 * 1;
            for (int var128 = 0; var128 < 3; var128 = var128 + 1) {
              int var129 = var127 + (var128 * 4);
              if (var129 < 10) {
                continue;
              } 
              if (var129 < 20) {
                for (int var130 = 0; var130 < 3; var130 = var130 + 1) {
                  int var132 = arg0[(((var3 * 500) + (var5 * 100)) + ((var129 - 10) * 10)) + ((var126 + (var130 * 4)) - 10)];
                  var7 = (((var4 * 45) + (var5 * 9)) + (var128 * 3)) + var130;
                  var2.data[var6] = var2.data[var6] + (var132 * arg1[var7]);
                  var8 = var8 + 1;
                }
              } else {
                break;
              }
            }
          }
          for (int var133 = 12; var133 < 22; var133 = var133 + 1) {
            var6 = (((var3 * 1452) + (var4 * 484)) + (var108 * var2.width)) + var133;
            int var134 = var133 * 1;
            int var135 = var108 * 1;
            for (int var136 = 0; var136 < 3; var136 = var136 + 1) {
              int var137 = var135 + (var136 * 4);
              if (var137 < 10) {
                continue;
              } 
              if (var137 < 20) {
                for (int var138 = 0; var138 < 3; var138 = var138 + 1) {
                  int var139 = var134 + (var138 * 4);
                  if (var139 < 10) {
                    continue;
                  } 
                  if (var139 < 20) {
                    int var140 = arg0[(((var3 * 500) + (var5 * 100)) + ((var137 - 10) * 10)) + (var139 - 10)];
                    var7 = (((var4 * 45) + (var5 * 9)) + (var136 * 3)) + var138;
                    var2.data[var6] = var2.data[var6] + (var140 * arg1[var7]);
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



