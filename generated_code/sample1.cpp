#include "mem_allocation.h"

void run_conv2d (void) {
  int var0 = 6;
  int var1 = 3;
  int var2 = (var0 - var1) + 1;
  int* var3 = conv_runtime::malloc((4 * var0) * var0);
  int* var4 = conv_runtime::malloc((4 * var1) * var1);
  int* var5 = conv_runtime::malloc((4 * var2) * var2);
  for (int var6 = 0; var6 < var0; var6 = var6 + 1) {
    for (int var7 = 0; var7 < var0; var7 = var7 + 1) {
      var3[(var6 * var0) + var7] = (var6 * var0) + var7;
    }
  }
  for (int var8 = 0; var8 < var1; var8 = var8 + 1) {
    for (int var9 = 0; var9 < var1; var9 = var9 + 1) {
      var4[(var8 * var1) + var9] = (var8 * var1) + var9;
    }
  }
  int var10 = var2;
  int var11 = var1;
  int var12 = var0;
  int* var13 = var5;
  int* var14 = var4;
  int* var15 = var3;
  for (int var16 = 0; var16 < var10; var16 = var16 + 1) {
    for (int var17 = 0; var17 < var10; var17 = var17 + 1) {
      var13[(var16 * var10) + var17] = 0;
      for (int var18 = 0; var18 < var11; var18 = var18 + 1) {
        for (int var19 = 0; var19 < var11; var19 = var19 + 1) {
          var13[(var16 * var10) + var17] = var13[(var16 * var10) + var17] + (var14[(var18 * var11) + var19] * var15[((var16 + var18) * var12) + (var17 + var19)]);
        }
      }
    }
  }
  conv_runtime::free(var3);
  conv_runtime::free(var4);
  conv_runtime::free(var5);
}

