struct TestOptions {
    int iw;
    int ih;
    int ww;
    int wh;
    int batch_size;
    int in_channels;
    int out_channels;
    int* stride;
    int padding_same;
    int* padding;
    int* dilation;
};