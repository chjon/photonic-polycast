#ifndef PPCAST_RENDERABLE_H
#define PPCAST_RENDERABLE_H

#include "Common.h"
#include "Ray.h"

namespace PPCast {
    struct HitInfo {
        float t = 0;
        glm::vec4 hitPoint;
        glm::vec4 normal;
    };
    
    class Renderable {
    public:
        virtual bool getIntersection(HitInfo& hitInfo, const Ray& r) const = 0;
    };
}

#endif