#ifndef PPCAST_RENDERABLE_H
#define PPCAST_RENDERABLE_H

#include "Common.h"
#include "Ray.h"

namespace PPCast {
    class Material;
    class Renderable {
    public:
        virtual bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const = 0;
    };
}

#endif