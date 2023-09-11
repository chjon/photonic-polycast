#ifndef PPCAST_GEOMETRY_H
#define PPCAST_GEOMETRY_H

#include "Common.h"
#include "Ray.cuh"

namespace PPCast {
    class Geometry {
    public:
        enum class Primitive: int {
            Sphere = 0,
            Cube   = 1,
        };
    
        __host__ __device__ static bool intersect      (HitInfo& hitInfo, Primitive p, const Ray& r, const Interval<float>& tRange);
        __host__ __device__ static bool intersectSphere(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange);
        __host__ __device__ static bool intersectCube  (HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange);
    };
}

#endif