#ifndef PPCAST_WORLD_H
#define PPCAST_WORLD_H

#include "Common.h"
#include "SceneNode.h"

namespace PPCast {
    class World {
    private:
        std::vector<Material> m_materials;

        // Just a vector for now
        // TODO: make this an octree or something
        std::vector<GeometryNode> m_geometry;

    public:
        World(std::vector<Material>&& materials, std::vector<GeometryNode>&& scene)
            : m_materials(materials)
            , m_geometry(scene)
        {}

        __host__ __device__ static bool getIntersection(
            HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange,
            const GeometryNode* geometry, size_t numGeometry
        );
        
        inline bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const {
            return getIntersection(
                hitInfo, r, tRange,
                m_geometry.data(), m_geometry.size()
            );
        }

        const std::vector<Material>& getMaterials() const { return m_materials; }
        const std::vector<GeometryNode>& getGeometry() const { return m_geometry; }
    };
}

#endif