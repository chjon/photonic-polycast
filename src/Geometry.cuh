#ifndef PPCAST_GEOMETRY_H
#define PPCAST_GEOMETRY_H

#include "Common.h"
#include "Ray.cuh"

namespace PPCast {
    class Geometry {
    private:
        /**
         * @brief Check whether a ray intersects with a sphere
         * 
         * @param hitInfo The output data structure in which to write information about intersections
         * @param r The ray to check for intersection
         * @param tRange The acceptable range of scaling values for the ray's direction vector
         * @return true iff there is an intersection within the interval  
         */
        __host__ __device__ static bool intersectSphere(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange);

        /**
         * @brief Check whether a ray intersects with a cube
         * 
         * @param hitInfo The output data structure in which to write information about intersections
         * @param r The ray to check for intersection
         * @param tRange The acceptable range of scaling values for the ray's direction vector
         * @return true iff there is an intersection within the interval  
         */
        __host__ __device__ static bool intersectCube  (HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange);

    public:
        /**
         * @brief Types of geometric primitives which can be checked for intersection
         * 
         */
        enum class Primitive: int {
            Sphere = 0,
            Cube   = 1,
        };
    
        /**
         * @brief Check whether a ray intersects with the geometry
         * 
         * @param hitInfo The output data structure in which to write information about intersections
         * @param p The type of primitive to check for intersection
         * @param r The ray to check for intersection
         * @param tRange The acceptable range of scaling values for the ray's direction vector
         * @return true iff there is an intersection within the interval 
         */
        __host__ __device__ static bool intersect(HitInfo& hitInfo, Primitive p, const Ray& r, const Interval<float>& tRange);
    };
}

#endif