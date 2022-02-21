#ifndef MEM_ALLOCATION_H
#define MEM_ALLOCATION_H

namespace conv_runtime {
    static int* malloc(int size) {
        return (int*)malloc(size);
    }
    static void free(int* ptr) {
        free(ptr);
    }
}

#endif