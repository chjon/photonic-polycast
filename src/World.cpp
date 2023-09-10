#include "World.h"

using namespace PPCast;

__host__ __device__ bool World::getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const {
    bool hit = false;
    for (const GeometryNode& node : m_geometry) {
        HitInfo tmpHitInfo;
        if (node.getIntersection(tmpHitInfo, r, tRange)) {
            hit = true;
            if (tmpHitInfo.t < hitInfo.t) hitInfo = tmpHitInfo;
        }
    }

    return hit;
}