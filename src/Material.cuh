#ifndef PPCAST_MATERIAL_H
#define PPCAST_MATERIAL_H

#include "Common.h"
#include "Ray.cuh"
#include "Types.cuh"

namespace PPCast {
    union RandomState;

    /**
     * @brief Types of materials -- each material type should have a corresponding scatter function
     * @note We are not using subclasses because the Material class needs to have constant size for array allocation
     */
    enum MaterialType: uint32_t {
        NormalDir  = 0, // Draw normal vector
        ReflectDir = 1, // Draw reflection vector
        RefractDir = 2, // Draw refraction vector
        Diffuse    = 3, // Diffuse scattering
        Lambertian = 4, // Lambertian diffuse scattering
        Reflective = 5, // Smooth scattering
        Refractive = 6, // Refraction + probabilistic reflection
    };

    /**
     * @brief Material parameters
     * 
     */
    struct MaterialData {
        /// @brief Material colour -- this is a multiplier for the colour components of
        /// reflected/transmitted light
        glm::vec3 attenuation = glm::vec3(1.f);
        
        /// @brief The distance that a reflected ray's direction vector can be randomly perturbed
        float fuzzFactor = 0.f;
        
        /// @brief The material's index of refraction
        float indexOfRefraction = 1.f;
    };

    class Material {
    public:
        /// @brief A unique identifier for the material
        const MaterialID id;

        /// @brief The type of material -- used to select a scatter function
        const MaterialType type;

        /// @brief Specific parameters for the material -- used by the scatter function
        MaterialData data;

        /**
         * @brief Construct a new Material object
         * 
         * @param matType The type of material
         * @param matData Additional parameters for the material
         */
        Material(MaterialType matType, const MaterialData& matData)
            : id(MaterialID::next())
            , type(matType)
            , data(matData)
        {}

        /**
         * @brief 
         * 
         * @param directionOut Output: the direction of the emitted ray
         * @param attenuation Output: the colour of the material
         * @param directionIn The direction of the incoming ray
         * @param hitInfo Information about how the ray hit the material
         * @param randomState State for random number generation
         * @return true if the material emits a ray 
         */
        __host__ __device__ bool scatter(
            glm::vec4& directionOut,
            glm::vec3& attenuation,
            const glm::vec4& directionIn,
            const HitInfo& hitInfo,
            RandomState& randomState
        ) const;
    };
}

#endif