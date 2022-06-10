#ifndef CONV_SCHEDULE_H
#define CONV_SCHEDULE_H
#include <assert.h>


namespace conv {

struct LoopSchedule {
    bool vectorized = false;
    bool unrolled = false;
    int parallel_collapse = 0;

    // tiling
    bool tiled = false;
    bool first = true; // first tiled loop
    bool last = true; // last tiled loop

    int bound; // upper bound
    int stride = 1; 

    // if this is an image loop does it come after 
    // a kernel loop for the same dim, and vice versa
    bool after = false;

    enum class loop_type {
        N,   // batches
        IC,  // in channels
        OC,  // out channels
        IMG,  
        KERNEL,
    };

    loop_type type;

    int dim; // 0 for height, 1 for width, etc.

    LoopSchedule(loop_type loop, int n) {
        bound = n;
        type = loop;
    }

    void parallelize(int collapse) {
        parallel_collapse = collapse;
    }

    void vectorize() {
        vectorized = true;
    }

    void unroll() {
        unrolled = true;
    }

    LoopSchedule* tile(int* dims, int n_subloops, LoopSchedule* subloops, int loop_len) {
        int total = 1;
        // std::cout << bound << std::endl;
        for (int i = 0; i < n_subloops; i++) {
            total *= dims[i];
            subloops[i] = LoopSchedule(type, loop_len / total * dims[i]);
            subloops[i].dim = dim;
            subloops[i].stride = loop_len / total;
            subloops[i].tiled = true;
            if (i != n_subloops - 1) {
                subloops[i].last = false;
            }
            if (i != 0) {
                subloops[i].first = false;
            }
        }
        return subloops;
    }
};

struct Schedule {

    int n_loops;
    int ndims;
    LoopSchedule* loops;

    Schedule(LoopSchedule* loop_arr, int nloops, int n_dims) {
        loops = loop_arr;
        n_loops = nloops;
        ndims = n_dims;
        int* img_found = new int[ndims];
        int* ker_found = new int[ndims];
        for (int i = 0; i < ndims; i++) {
            img_found[i] = 0;
            ker_found[i] = 0;
        }
        
        for (int i = 0; i < n_loops; i = i + 1) {
            LoopSchedule loop = loop_arr[i];
            if (!loop.last) continue;
            if (loop.type == LoopSchedule::loop_type::IMG) {
                if (ker_found[loop.dim]) {
                    loop_arr[i].after = 1;
                }
                img_found[loop.dim] = 1;
            } else if (loop.type == LoopSchedule::loop_type::KERNEL) {
                if (img_found[loop.dim]) {
                    loop_arr[i].after = 1;
                }
                ker_found[loop.dim] = 1;
            }
        }
        delete[] img_found;
        delete[] ker_found;
    }

  
};


}
#endif