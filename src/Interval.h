#ifndef PPCAST_INTERVAL_H
#define PPCAST_INTERVAL_H

#include <algorithm>
#include "Common.h"

namespace PPCast {
    template <typename T>
    class Interval {
    public:
        T lower;
        T upper;
        bool minOpen;
        bool maxOpen;

        __host__ __device__ Interval(T min_, T max_, bool minOpen_, bool maxOpen_)
            : lower(
                #ifdef __CUDA_ARCH__
                min(min_, max_)
                #else
                std::min(min_, max_)                
                #endif
            )
            , upper(
                #ifdef __CUDA_ARCH__
                max(min_, max_)
                #else              
                std::max(min_, max_)
                #endif
            )
            , minOpen(minOpen_)
            , maxOpen(maxOpen_)
        {}

        __host__ __device__ bool contains(T val) const {
            return !(
                (val < lower) ||
                (val > upper) ||
                (minOpen && val == lower) ||
                (maxOpen && val == upper)
            );
        }
    };
}

#endif