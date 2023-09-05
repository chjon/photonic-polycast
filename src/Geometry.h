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
    
        static bool intersect      (float& t, glm::vec4& normal, Primitive p, const Ray& r);
        static bool intersectSphere(float& t, glm::vec4& normal, const Ray& r);
        static bool intersectCube  (float& t, glm::vec4& normal, const Ray& r);
    };
}

#endif