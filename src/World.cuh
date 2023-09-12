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

        __host__ World(const std::vector<Material>& mats, const std::vector<GeometryNode>& geom)
            : materials(mats)
            , geometry (geom)
        {}

        __device__ World(const VectorRef<Material>& mats, const VectorRef<GeometryNode>& geom)
            : materials(mats)
            , geometry (geom)
        {}
        
        __host__ __device__ bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const;
    };
}

#endif