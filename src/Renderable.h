#ifndef PPCAST_RENDERABLE_H
#define PPCAST_RENDERABLE_H

#include "Common.h"
#include "Ray.h"

namespace PPCast {
    class Material {
    public:
        virtual bool scatter(
            glm::vec4& directionOut,
            glm::vec3& attenuation,
            const glm::vec4& directionIn,
            const glm::vec4& normal
        ) const = 0;
    };

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

    class MaterialNormal: public Material {
    public:
        bool scatter(glm::vec4&, glm::vec3& attenuation, const glm::vec4&, const glm::vec4& normal) const override {
            attenuation = 0.5f * (glm::normalize(normal.xyz()) + glm::vec3(1));
            return false;
        }
    };

    class MaterialDiffuse: public Material {
    protected:
        glm::vec3 m_attenuation;
    
    public:
        MaterialDiffuse(const glm::vec3& colour)
            : m_attenuation(colour)
        {}

        bool scatter(glm::vec4& directionOut, glm::vec3& attenuation, const glm::vec4&, const glm::vec4& normal) const override {
            directionOut = randomOnHemisphere(normal);
            attenuation = m_attenuation;
            return true;
        }
    };

    class MaterialLambertian: public Material {
    protected:
        glm::vec3 m_attenuation;
    
    public:
        MaterialLambertian(const glm::vec3& colour)
            : m_attenuation(colour)
        {}

        bool scatter(glm::vec4& directionOut, glm::vec3& attenuation, const glm::vec4&, const glm::vec4& normal) const override {
            directionOut = glm::normalize(normal) + glm::vec4(randomUnitVector(), 0);
            attenuation = m_attenuation;
            return true;
        }
    };

    class MaterialMetal: public Material {
    protected:
        glm::vec3 m_attenuation;
    
    public:
        MaterialMetal(const glm::vec3& colour)
            : m_attenuation(colour)
        {}

        bool scatter(glm::vec4& directionOut, glm::vec3& attenuation, const glm::vec4& directionIn, const glm::vec4& normal) const override {
            directionOut = directionIn - 2.f * glm::dot(directionIn, normal) * glm::normalize(normal);
            attenuation = m_attenuation;
            return true;
        }
    };
}

#endif