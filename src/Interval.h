#ifndef PPCAST_INTERVAL_H
#define PPCAST_INTERVAL_H

#include <algorithm>
#include "Common.h"

namespace PPCast {
    template <typename T>
    class Interval {
    public:
        T min;
        T max;
        bool minOpen;
        bool maxOpen;

        __host__ __device__ Interval(T min_, T max_, bool minOpen_, bool maxOpen_)
            : min(std::min(min_, max_))
            , max(std::max(min_, max_))
            , minOpen(minOpen_)
            , maxOpen(maxOpen_)
        {}

        __host__ __device__ bool contains(T val) const {
            return !(
                (val < min) ||
                (val > max) ||
                (minOpen && val == min) ||
                (maxOpen && val == max)
            );
        }
    };
}

#endif