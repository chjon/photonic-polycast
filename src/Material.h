#ifndef PPCAST_MATERIAL_H
#define PPCAST_MATERIAL_H

#include "Common.h"
#include "Ray.h"
#include "Types.h"

namespace PPCast {
    // Note: not using subclasses because the Material class needs to have constant size for array allocation
    enum MaterialType: uint32_t {
        NormalDir  = 0, // Draw normal vector
        ReflectDir = 1, // Draw reflection vector
        RefractDir = 2, // Draw refraction vector
        Diffuse    = 3, // Diffuse scattering
        Lambertian = 4, // Lambertian diffuse scattering
        Reflective = 5, // Smooth scattering
        Refractive = 6, // Refraction + probabilistic reflection
    };

    struct MaterialData {
        glm::vec3 attenuation       = glm::vec3(1.f);
        float     fuzzFactor        = 0.f;
        float     indexOfRefraction = 1.f;
    };

    class Material {
    public:
        const MaterialID id;
        const MaterialType type;
        MaterialData data;

        Material(MaterialType matType, const MaterialData& matData)
            : id(MaterialID::next())
            , type(matType)
            , data(matData)
        {}

        bool scatter(
            glm::vec4& directionOut,
            glm::vec3& attenuation,
            const glm::vec4& directionIn,
            const HitInfo& hitInfo
        ) const;
    };
}

#endif