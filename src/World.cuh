#ifndef PPCAST_WORLD_H
#define PPCAST_WORLD_H

#include "Common.h"
#include "CudaSerializable.cuh"
#include "SceneNode.cuh"
#include "Types.cuh"

namespace PPCast {
    /**
     * @brief This class represents the scene to render
     * 
     */
    class World {
    public:
        /// @brief The materials in the scene
        const VectorRef<Material> materials;

        /// @brief The geometry in the scene
        const VectorRef<GeometryNode> geometry;

        /**
         * @brief Construct a new World object
         * 
         * @param mats the materials in the scene
         * @param geom the geometry in the scene
         */
        __host__ World(const std::vector<Material>& mats, const std::vector<GeometryNode>& geom)
            : materials(mats)
            , geometry (geom)
        {}

        /**
         * @brief Construct a new World object
         * 
         * @param mats the materials in the scene
         * @param geom the geometry in the scene
         */
        __device__ World(const VectorRef<Material>& mats, const VectorRef<GeometryNode>& geom)
            : materials(mats)
            , geometry (geom)
        {}
        
        /**
         * @brief Check whether a ray intersects with any of the objects in the scene
         * 
         * @param hitInfo Output: information about how the ray hits an object
         * @param r The ray to check
         * @param tRange The valid range for multiples of the ray's direction vector
         * @return true if the ray intersects with any object
         */
        __host__ __device__ bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const;
    };
}

#endif