#include "World.cuh"

using namespace PPCast;

__host__ __device__ bool World::getIntersection(
    HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange,
    const GeometryNode* geometry, size_t numGeometry
) {
    bool hit = false;
    for (unsigned int i = 0; i < numGeometry; ++i) {
        const GeometryNode& node = geometry[i];
        HitInfo tmpHitInfo;
        if (node.getIntersection(tmpHitInfo, r, tRange)) {
            hit = true;
            if (tmpHitInfo.t < hitInfo.t) hitInfo = tmpHitInfo;
        }
    }

    return hit;
}