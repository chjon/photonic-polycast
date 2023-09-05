#ifndef PPCAST_CAMERA_H
#define PPCAST_CAMERA_H

#include <png++/png.hpp>
#include "Common.h"
#include "SceneNode.h"

namespace PPCast {
    class Camera {
    private:
        // Camera position and orientation
        glm::vec3 m_pos;
        glm::vec3 m_centre;
        glm::vec3 m_up;

        // Camera settings
        float m_fovy;
        float m_aspect;

        uint32_t m_jitter;

        glm::mat4x4 getView() const { return glm::lookAt(m_pos, m_centre, m_up); }

    public:
        /**
         * @brief Default camera constructor -- constructs camera using commandline options
         * 
         */
        Camera(const glm::vec3& pos, const glm::vec3& centre, const glm::vec3& up);

        png::image<png::rgb_pixel> render(const std::vector<GeometryNode>& scene, uint32_t width, uint32_t height) const;
    };
}

#endif