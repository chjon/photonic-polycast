#ifndef PPCAST_CAMERA_H
#define PPCAST_CAMERA_H

#include <cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <png++/png.hpp>

namespace PPCast {
    class Camera {
    private:
        // Camera position and orientation
        glm::vec3 m_pos;
        glm::vec3 m_centre;
        glm::vec3 m_up;

        // Camera settings
        float m_fov_y;
        float m_aspect;
        float m_nearPlane;
        float m_farPlane;

        glm::mat4x4 getView() { return glm::lookAt(m_pos, m_centre, m_up); }
        glm::mat4x4 getProj() { return glm::perspective(m_fov_y, m_aspect, m_nearPlane, m_farPlane); }

    public:
        /**
         * @brief Default camera constructor -- constructs camera using commandline options
         * 
         */
        Camera(const glm::vec3& pos, const glm::vec3& centre, const glm::vec3& up);

        png::image<png::rgb_pixel> render(uint32_t width, uint32_t height) const;
    };
}

#endif