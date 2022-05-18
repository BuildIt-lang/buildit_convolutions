#ifndef CONV_SCHEDULE_H
#define CONV_SCHEDULE_H
#include <assert.h>

namespace conv {

struct LoopSchedule {
    bool vectorized = false;
    bool unrolled = false;
    int bound;
    int parallel_collapse = 0;
    bool after = false;

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

    LoopSchedule* tile(int* dims, int n_subloops) {
        LoopSchedule* subloops = nullptr;
        int total = 1;
        for (int i = 0; i < n_subloops; i++) {
            total *= dims[i];
            subloops[i] = LoopSchedule(type, dims[i]);
        }
        if (total != bound) {
            std::cout << "Error: invalid tile sizes" << std::endl;
            assert(false);
        }
        return subloops;
    }
};

struct Schedule {

    int n_loops = 7;
    LoopSchedule* loops;

  
};


}
#endif