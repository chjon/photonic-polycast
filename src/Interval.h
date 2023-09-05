#ifndef PPCAST_INTERVAL_H
#define PPCAST_INTERVAL_H

#include <algorithm>

namespace PPCast {
    template <typename T>
    class Interval {
    public:
        T min;
        T max;
        bool minOpen;
        bool maxOpen;

        Interval(T min_, T max_, bool minOpen_, bool maxOpen_)
            : min(std::min(min_, max_))
            , max(std::max(min_, max_))
            , minOpen(minOpen_)
            , maxOpen(maxOpen_)
        {}

        bool contains(T val) const {
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