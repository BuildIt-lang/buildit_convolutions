namespace conv_runtime {
    static void* malloc(int size) {
        return malloc(size);
    }
    static void free(int* ptr) {
        free(ptr);
    }
}