#ifndef RUNTIME_FUNCTIONS_H
#define RUNTIME_FUNCTIONS_H

#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

namespace conv_runtime {
    int* conv_malloc(int size) {
        return (int*) malloc(size);
    }
    int* conv_calloc(int num, int size) {
        return (int*) calloc(num, size);
    }
    void conv_free(int* ptr) {
        free(ptr);
    }

    // timer from graphit
    static struct timeval start_time_;
    static struct timeval elapsed_time_;

    static void start_timer(void) {
        gettimeofday(&start_time_, NULL);
    }
    static float stop_timer(void) {
        struct timeval stop_time_;
        gettimeofday(&stop_time_, NULL);
        timersub(&stop_time_, &start_time_, &elapsed_time_);
        return elapsed_time_.tv_sec + elapsed_time_.tv_usec/1e6f;
    }
    static void print_time(float x) {
        std::cout << x << std::endl;
    }

    void print_matrix(int* m, int size) {
        for (int i = 0; i < size; i = i + 1) {
            for (int j = 0; j < size; j = j + 1) {
                std::cout << m[i * size + j] << " ";
            }
            std::cout << "\n";
        }
    }
}

#endif