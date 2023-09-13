#ifndef PPCAST_INTERVAL_H
#define PPCAST_INTERVAL_H

#include <algorithm>
#include "Common.h"

namespace PPCast {
    /**
     * @brief An interval of values
     * 
     * @tparam T the type of value in the interval (must have a total order)
     */
    template <typename T>
    class Interval {
    public:
        /// @brief The lower end of the interval
        T lower;

        /// @brief The upper end of the interval
        T upper;

        /// @brief Whether the lower end of the interval is an acceptable value
        bool minOpen;

        /// @brief Whether the upper end of the interval is an acceptable value
        bool maxOpen;

        /**
         * @brief Construct an Interval object
         * 
         * @param min_ The lower end of the interval
         * @param max_ The upper end of the interval
         * @param minOpen_ Whether the lower end of the interval is an acceptable value
         * @param maxOpen_ Whether the upper end of the interval is an acceptable value
         */
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

        /**
         * @brief Determine whether a value lies within the interval
         * 
         * @param val The value to check
         * @return true if the value lies within the interval 
         */
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