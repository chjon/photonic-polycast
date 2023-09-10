#ifndef PPCAST_WORLD_H
#define PPCAST_WORLD_H

#include "Common.h"
#include "Renderable.h"
#include "SceneNode.h"

namespace PPCast {
    class World : public Renderable {
    private:
        std::vector<Material> m_materials;

        // Just a vector for now
        // TODO: make this an octree or something
        std::vector<GeometryNode> m_geometry;

    public:
        World(std::vector<Material>&& materials, std::vector<GeometryNode>&& scene)
            : Renderable()
            , m_materials(materials)
            , m_geometry(scene)
        {}

        __host__ __device__ bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const override;
        const std::vector<Material>& getMaterials() const { return m_materials; }
        const std::vector<GeometryNode>& getGeometry() const { return m_geometry; }
    };
}

#endif