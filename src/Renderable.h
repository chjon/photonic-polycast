#ifndef PPCAST_RENDERABLE_H
#define PPCAST_RENDERABLE_H

#include "Common.h"
#include "Ray.h"

namespace PPCast {
    class Material;

    struct HitInfo {
        float t = std::numeric_limits<float>::max();
        glm::vec4 hitPoint;
        glm::vec4 normal;
        std::shared_ptr<Material> material;
    };

    class Renderable {
    public:
        std::shared_ptr<Material> material;
        Renderable(const std::shared_ptr<Material>& mat) : material(mat) {}

        virtual bool getIntersection(HitInfo& hitInfo, const Ray& r) const = 0;
    };
}

#endif