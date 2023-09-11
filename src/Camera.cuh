#ifndef PPCAST_CAMERA_H
#define PPCAST_CAMERA_H

#include "Common.h"
#include "Ray.cuh"
#include "SceneNode.cuh"
#include "World.cuh"

namespace PPCast {
    class Image;
    class RandomState;
    class World;

    class Camera {
    private:
        // Derived values -- cached for generating rays
        glm::vec3 m_pixel_dx;
        glm::vec3 m_pixel_dy;
        glm::vec3 m_pixel_topLeft;
        glm::mat4 m_v2w;
        float     m_defocusRadius;

        bool renderImageCPU(Image& image, const World& scene) const;
        bool renderImageGPU(Image& image, const World& scene) const;

    public:
        // Camera position and orientation
        glm::vec3 pos;
        glm::vec3 centre;
        glm::vec3 up;

        // Camera settings
        float    vfov;
        float    dofAngle;
        float    focalDist;

        // Image dimensions
        uint32_t width;
        uint32_t height;

        // Image quality
        uint32_t raysPerPx;
        uint32_t maxBounces;

        // Random number generation
        uint32_t seed;

        /**
         * @brief Default camera constructor -- constructs camera using commandline options
         * 
         */
        Camera();

        void initialize(uint32_t width, uint32_t height);
        bool renderImage(Image& image, const World& scene) const;

        __host__ __device__ Ray generateRay(uint32_t x, uint32_t y, RandomState& randomState) const;

        __host__ __device__ static glm::vec3 raycast(
            const Ray& ray, Interval<float>&& tRange,
            const PPCast::Material* materials, size_t numMaterials,
            const PPCast::GeometryNode* geometry, size_t numGeometry,
            unsigned int maxDepth, RandomState& randomState
        );

        static inline glm::vec3 raycast(
            const Ray& ray, Interval<float>&& tRange,
            const World& world,
            unsigned int maxDepth, RandomState& randomState
        ) {
            const std::vector<Material>& materials = world.getMaterials();
            const std::vector<GeometryNode>& geometry = world.getGeometry();
            return raycast(
                ray, std::move(tRange),
                materials.data(), materials.size(),
                geometry.data(), geometry.size(),
                maxDepth, randomState
            );
        }

        __host__ __device__ glm::vec3 renderPixel(
            uint32_t x, uint32_t y,
            const PPCast::Material* materials, size_t numMaterials,
            const PPCast::GeometryNode* geometry, size_t numGeometry,
            RandomState& randomState
        ) const;

        inline glm::vec3 renderPixel(
            uint32_t x, uint32_t y,
            const World& world,
            RandomState& randomState
        ) const {
            const std::vector<Material>& materials = world.getMaterials();
            const std::vector<GeometryNode>& geometry = world.getGeometry();
            return renderPixel(
                x, y,
                materials.data(), materials.size(),
                geometry.data(), geometry.size(),
                randomState
            );
        }
    };
}

#endif