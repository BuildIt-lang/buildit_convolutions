namespace conv_runtime {

template <typename T>
struct TensorT {
    int width;
    int height;
    T* data;
    void print() {
        for (int i = 0; i < height; i = i + 1) {
            for (int j = 0; j < width; j = j + 1) {
                std::cout << data[i * width + j] << " ";
            }
            std::cout << "\n";
        }
    }
};

}