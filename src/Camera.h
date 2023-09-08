#ifndef PPCAST_CAMERA_H
#define PPCAST_CAMERA_H

#include <png++/png.hpp>
#include "Common.h"
#include "Ray.h"
#include "SceneNode.h"

namespace PPCast {
    class Camera {
    private:
        // Derived values -- cached for generating rays
        glm::vec3 m_pixel_dx;
        glm::vec3 m_pixel_dy;
        glm::vec3 m_pixel_topLeft;
        glm::mat4 m_v2w;

        Ray generateRay(uint32_t x, uint32_t y) const;
        glm::vec3 renderPixel(uint32_t x, uint32_t y, const std::vector<GeometryNode>& scene) const;

    public:
        // Camera position and orientation
        glm::vec3 pos;
        glm::vec3 centre;
        glm::vec3 up;

        // Camera settings
        float    fovy;

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

        png::image<png::rgb_pixel> renderImage(const std::vector<GeometryNode>& scene) const;
    };
}

#endif