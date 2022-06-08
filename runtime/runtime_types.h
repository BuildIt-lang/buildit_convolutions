namespace conv_runtime {

template <typename T>
struct ImageT {
    int batch_size;
    int in_channels;
    int ndims;
    int* dims;
    T* data;
    int64_t mult_cnt;
    void print() {
        for (int i = 0; i < dims[0]; i = i + 1) {
            for (int j = 0; j < dims[1]; j = j + 1) {
                std::cout << data[i * dims[1] + j] << " ";
            }
            std::cout << "\n";
        }
    }
};

struct PaddingT {
    bool is_same;
    int* values;
    PaddingT(char* inp_type) {
        if (strcmp("same", inp_type) != 0) {
            std::cout << "Invalid padding type." << std::endl;
            assert(false);
        }
        is_same = true;
    }
    PaddingT(int* inp_values) {
        is_same = false;
        values = inp_values;
    }
};

struct ConvOptions {
    int* stride;
    PaddingT padding;
    int* dilation;
    int groups;
};

}
