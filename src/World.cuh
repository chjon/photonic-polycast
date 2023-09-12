#ifndef PPCAST_WORLD_H
#define PPCAST_WORLD_H

#include "Common.h"
#include "CudaSerializable.cuh"
#include "SceneNode.cuh"
#include "Types.cuh"

namespace PPCast {
    class World {
    public:
        const VectorRef<Material> materials;
        const VectorRef<GeometryNode> geometry;

        World(const std::vector<Material>& mats, const std::vector<GeometryNode>& geom)
            : materials(mats)
            , geometry (geom)
        {}

        __host__ __device__ static bool getIntersection(
            HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange,
            const VectorRef<GeometryNode>& geometry
        );
        
        inline bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const {
            return getIntersection(hitInfo, r, tRange, geometry);
        }
    };
}

#endif