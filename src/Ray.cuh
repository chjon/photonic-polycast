#ifndef PPCAST_RAY_H
#define PPCAST_RAY_H

#include "Common.h"
#include "Interval.cuh"
#include "Types.cuh"

namespace PPCast {
    class Material;

    /**
     * @brief Information about where a ray hits an object
     * 
     */
    struct HitInfo {
        /// @brief The multiple of the ray's direction vector where the ray intersects the object
        float t = std::numeric_limits<float>::max();

        /// @brief The location of the hit
        glm::vec4 hitPoint;

        /// @brief The object's normal vector at the hit position
        glm::vec4 normal;

        /// @brief Whether the ray hit the outside of the object
        bool hitOutside;

        /// @brief The ID of the object's associated material
        MaterialID materialID;
    };

    /**
     * @brief A class representing the path of a single photon
     * 
     */
    class Ray {
    private:
        /// @brief The ray's origin position
        glm::vec4 m_origin;

        /// @brief The ray's direction
        glm::vec4 m_direction;

        /// @brief The ray's sample time
        float m_time;

    public:
        /**
         * @brief Construct a new Ray
         * 
         * @param origin The origin position of the ray
         * @param direction The direction of the ray
         * @param time The time at which the ray was sampled
         */
        __host__ __device__ Ray(const glm::vec3& origin, const glm::vec3& direction, const float time)
            : m_origin   (origin, 1)
            , m_direction(direction, 0)
            , m_time     (time)
        {}
        
        /**
         * @brief Construct a new Ray
         * 
         * @param origin The origin position of the ray
         * @param direction The direction of the ray
         */
        __host__ __device__ Ray(const glm::vec4& origin, const glm::vec4& direction, const float time)
            : m_origin   (origin)
            , m_direction(direction)
            , m_time     (time)
        {}

        /**
         * @brief Get the ray's origin location
         * 
         * @return The ray's origin location 
         */
        __host__ __device__ const glm::vec4& origin   () const { return m_origin; }

        /**
         * @brief Get the ray's direction
         * 
         * @return The ray's direction
         */
        __host__ __device__ const glm::vec4& direction() const { return m_direction; }

        /**
         * @brief Get the ray's sample time
         * 
         * @return The ray's sample time
         */
        __host__ __device__ const float& time() const { return m_time; }

        /**
         * @brief Get the position of the ray for given scaling value
         * 
         * @param t The multiplier for the ray's direction vector
         * @return The position of the ray at the given scaling value
         */
        __host__ __device__ const glm::vec4 at(const float t) const { return m_origin + t * m_direction; }

    private:
        /**
         * @brief Output the ray
         * 
         * @param os The output stream
         * @param r The ray to output
         * @return The output stream -- returned for chaining calls to @code{operator<<}
         */
        friend std::ostream& operator<<(std::ostream &os, const Ray& r) {
            return os << "(o" << r.m_origin << ", d" << r.m_direction << ")";
        }
    };
}

#endif