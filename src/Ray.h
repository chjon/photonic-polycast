#ifndef PPCAST_RAY_H
#define PPCAST_RAY_H

#include <glm/vec3.hpp>

namespace PPCast {
    class Ray {
    private:
        glm::vec3 m_origin;
        glm::vec3 m_direction;

    public:
        Ray(const glm::vec3& origin, const glm::vec3& direction)
            : m_origin   (origin)
            , m_direction(glm::normalize(direction))
        {}

        const glm::vec3 origin   () const { return m_origin; }
        const glm::vec3 direction() const { return m_direction; }

        const glm::vec3 at(float t) const { return m_origin + t * m_direction; }
    };
}

#endif