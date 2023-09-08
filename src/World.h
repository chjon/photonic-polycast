#ifndef PPCAST_WORLD_H
#define PPCAST_WORLD_H

#include "Common.h"
#include "Renderable.h"
#include "SceneNode.h"

namespace PPCast {
    class World : public Renderable {
    private:
        // Just a vector for now
        // TODO: make this an octree or something
        std::vector<GeometryNode> m_geometry;

    public:
        World(std::vector<GeometryNode>&& scene) : Renderable(), m_geometry(scene) {}

        bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const override;
    };
}

#endif