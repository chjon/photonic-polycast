#ifndef PPCAST_RAY_H
#define PPCAST_RAY_H

#include "Common.h"
#include "Interval.cuh"
#include "Types.cuh"

namespace PPCast {
    class Material;

    struct HitInfo {
        float t = std::numeric_limits<float>::max();
        glm::vec4 hitPoint;
        glm::vec4 normal;
        bool hitOutside;
        MaterialID materialID;
    };

    class Ray {
    private:
        glm::vec4 m_origin;
        glm::vec4 m_direction;

    public:
        __host__ __device__ Ray(const glm::vec3& origin, const glm::vec3& direction)
            : m_origin   (origin, 1)
            , m_direction(direction, 0)
        {}
        
        __host__ __device__ Ray(const glm::vec4& origin, const glm::vec4& direction)
            : m_origin   (origin)
            , m_direction(direction)
        {}

        __host__ __device__ const glm::vec4& origin   () const { return m_origin; }
        __host__ __device__ const glm::vec4& direction() const { return m_direction; }

        __host__ __device__ const glm::vec4 at(const float t) const { return m_origin + t * m_direction; }

    private:
        friend std::ostream& operator<<(std::ostream &os, const Ray& r) {
            return os << "(o" << r.m_origin << ", d" << r.m_direction << ")";
        }
    };
}

#endif