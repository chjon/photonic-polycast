#ifndef PPCAST_MATERIAL_H
#define PPCAST_MATERIAL_H

#include "Common.h"
#include "Ray.h"
#include "Types.h"

namespace PPCast {
    class Material {
    public:
        const MaterialID id;

        Material() : id(MaterialID::next()) {}

        virtual bool scatter(
            glm::vec4& directionOut,
            glm::vec3& attenuation,
            const glm::vec4& directionIn,
            const HitInfo& hitInfo
        ) const = 0;
    };

    /////////////////////////////
    // Materials for debugging //
    /////////////////////////////

    class MaterialNormal: public Material {
    public:
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const HitInfo&) const override;
    };

    class MaterialReflDir: public Material {
    public:
        MaterialReflDir() {}
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const HitInfo&) const override;
    };

    class MaterialRefrDir: public Material {
    protected:
        float m_refractiveIndex;
    
    public:
        MaterialRefrDir(const float refractiveIndex) : m_refractiveIndex(refractiveIndex) {}
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const HitInfo&) const override;
    };

    //////////////////////
    // Actual materials //
    //////////////////////

    class MaterialDiffuse: public Material {
    protected:
        glm::vec3 m_attenuation;
    
    public:
        MaterialDiffuse(const glm::vec3& colour) : m_attenuation(colour) {}
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const HitInfo&) const override;
    };

    class MaterialLambertian: public Material {
    protected:
        glm::vec3 m_attenuation;
    
    public:
        MaterialLambertian(const glm::vec3& colour) : m_attenuation(colour) {}
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const HitInfo&) const override;
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
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const HitInfo&) const override;
    };

    class MaterialRefractive: public Material {
    protected:
        glm::vec3 m_attenuation;
        float     m_refractiveIndex;
    
    public:
        MaterialRefractive(const glm::vec3& colour, const float refractiveIndex)
            : m_attenuation(colour)
            , m_refractiveIndex(refractiveIndex)
        {}
        bool scatter(glm::vec4&, glm::vec3&, const glm::vec4&, const HitInfo&) const override;
    };
}

#endif