#ifndef PPCAST_MATERIAL_H
#define PPCAST_MATERIAL_H

#include "Common.h"

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

    class MaterialNormal: public Material {
    public:
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const glm::vec4&) const override;
    };

    class MaterialDiffuse: public Material {
    protected:
        glm::vec3 m_attenuation;
    
    public:
        MaterialDiffuse(const glm::vec3& colour) : m_attenuation(colour) {}
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const glm::vec4&) const override;
    };

    class MaterialLambertian: public Material {
    protected:
        glm::vec3 m_attenuation;
    
    public:
        MaterialLambertian(const glm::vec3& colour) : m_attenuation(colour) {}
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const glm::vec4&) const override;
    };

    class MaterialMetal: public Material {
    protected:
        glm::vec3 m_attenuation;
        float     m_fuzzFactor;
    
    public:
        MaterialMetal(const glm::vec3& colour, const float fuzzFactor = 0.f)
            : m_attenuation(colour)
            , m_fuzzFactor(fuzzFactor)
        {}
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const glm::vec4&) const override;
    };
}

#endif