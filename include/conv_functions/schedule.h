#ifndef CONV_SCHEDULE_H
#define CONV_SCHEDULE_H
#include <assert.h>


namespace conv {

struct LoopSchedule {
    bool vectorized = false;
    bool unrolled = false;
    bool tiled = false;
    bool first = true;
    bool last = true;
    int bound;
    int parallel_collapse = 0;
    bool after = false;
    int stride = 1;

    enum class loop_type {
        N,   // batches
        IC,  // in channels
        OC,  // out channels
        IH,  // image height
        IW,  // image width
        KH,  // kernel height
        KW   // kernel width
    };

    loop_type type;

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
            subloops[i].stride = loop_len / total;
            subloops[i].tiled = true;
            if (i != n_subloops - 1) {
                subloops[i].last = false;
            }
            if (i != 0) {
                subloops[i].first = false;
            }
        }
        // if (total != bound) {
        //     std::cout << "Error: invalid tile sizes" << std::endl;
        //     assert(false);
        // }
        return subloops;
    }
};

struct Schedule {

    int n_loops;
    LoopSchedule* loops;

    Schedule(LoopSchedule* loop_arr, int n) {
        loops = loop_arr;
        n_loops = n;
        bool found_iw = false;
        bool found_ih = false;
        bool found_kh = false;
        bool found_kw = false;
        
        for (int i = 0; i < n_loops; i = i + 1) {
            LoopSchedule loop = loop_arr[i];
            if (!loop.last) continue;
            if (loop.type == LoopSchedule::loop_type::IW) {
                if (found_kw) {
                    loop_arr[i].after = true;
                }
                found_iw = true;
            } else if (loop.type == LoopSchedule::loop_type::IH) {
                if (found_kh) {
                    loop_arr[i].after = true;
                }
                found_ih = true;
            } else if (loop.type == LoopSchedule::loop_type::KW) {
                if (found_iw) {
                    loop_arr[i].after = true;
                }
                found_kw = true;
            } else if (loop.type == LoopSchedule::loop_type::KH) {
                if (found_ih) {
                    loop_arr[i].after = true;
                }
                found_kh = true;
            }
        }
    }

  
};


}
#endif