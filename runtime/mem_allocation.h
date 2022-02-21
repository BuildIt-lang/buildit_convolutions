#ifndef MEM_ALLOCATION_H
#define MEM_ALLOCATION_H

#include "stdlib.h"

namespace conv_runtime {
    int* conv_malloc(int size) {
        return (int*) malloc(size);
    }
    void conv_free(int* ptr) {
        free(ptr);
    }
}

#endif