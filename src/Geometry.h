#ifndef PPCAST_GEOMETRY_H
#define PPCAST_GEOMETRY_H

#include "Common.h"
#include "Ray.h"

namespace PPCast {
    class Geometry {
    public:
        enum class Primitive: int {
            Sphere = 0,
            Cube   = 1,
        };
    
        static bool intersect      (HitInfo& hitInfo, Primitive p, const Ray& r);
        static bool intersectSphere(HitInfo& hitInfo, const Ray& r);
        static bool intersectCube  (HitInfo& hitInfo, const Ray& r);
    };
}

#endif