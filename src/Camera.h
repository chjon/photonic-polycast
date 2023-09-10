#ifndef PPCAST_CAMERA_H
#define PPCAST_CAMERA_H

#include "Common.h"
#include "Ray.h"
#include "SceneNode.h"

namespace PPCast {
    class Image;
    class World;

    class Camera {
    private:
        // Derived values -- cached for generating rays
        glm::vec3 m_pixel_dx;
        glm::vec3 m_pixel_dy;
        glm::vec3 m_pixel_topLeft;
        glm::mat4 m_v2w;
        float     m_defocusRadius;

        __host__ __device__ Ray generateRay(uint32_t x, uint32_t y) const;
        glm::vec3 renderPixel(uint32_t x, uint32_t y, const World& scene) const;
        static glm::vec3 raycast(const Ray& ray, Interval<float>&& tRange, const World& world, unsigned int maxDepth);

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

        /**
         * @brief Default camera constructor -- constructs camera using commandline options
         * 
         */
        Camera();

        void initialize(uint32_t width, uint32_t height);
        bool renderImage(Image& image, const World& scene) const;
    };
}

#endif