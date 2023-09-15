#ifndef PPCAST_SCENENODE_H
#define PPCAST_SCENENODE_H

#include "Common.h"
#include "Geometry.cuh"
#include "Material.cuh"
#include "Curve.cuh"

namespace PPCast {
    /**
     * @brief A class representing a node in the scene
     * 
     */
    class SceneNode {
    protected:
        /// @brief The node's transformation matrix
        glm::mat4x4 transform;

        /// @brief The inverse of the node's transformation matrix
        glm::mat4x4 invtransform;

        /// @brief The node's motion curve
        MotionCurve motionCurve;

    public:
        /**
         * @brief Construct a new Scene Node object with no transformation
         * 
         */
        SceneNode(const MotionCurve& motionCurve_)
            : transform(1.f)
            , invtransform(1.f)
            , motionCurve(motionCurve_)
        {}

        //////////
        // Scaling
        inline SceneNode& scale(const glm::vec3& scaleFactor) {
            transform    = glm::scale(glm::mat4(1), scaleFactor) * transform;
            invtransform = glm::inverse(transform);
            return *this;
        }

        inline SceneNode& scale (const float scaleFactor) { return scale(glm::vec3(scaleFactor, scaleFactor, scaleFactor)); }
        inline SceneNode& scaleX(const float scaleFactor) { return scale(glm::vec3(scaleFactor, 1, 1)); }
        inline SceneNode& scaleY(const float scaleFactor) { return scale(glm::vec3(1, scaleFactor, 1)); }
        inline SceneNode& scaleZ(const float scaleFactor) { return scale(glm::vec3(1, 1, scaleFactor)); }

        ///////////////
        // Translations
        inline SceneNode& translate(const glm::vec3& distance) {
            transform    = glm::translate(glm::mat4(1), distance) * transform;
            invtransform = glm::inverse(transform);
            return *this;
        }
        inline SceneNode& translateX(const float distance) { return translate(glm::vec3(distance, 0, 0)); }
        inline SceneNode& translateY(const float distance) { return translate(glm::vec3(0, distance, 0)); }
        inline SceneNode& translateZ(const float distance) { return translate(glm::vec3(0, 0, distance)); }

        ////////////
        // Rotations

        inline SceneNode& rotate(const float angle_rad, const glm::vec3& axis) {
            transform    = glm::rotate(glm::mat4(1), angle_rad, axis) * transform;
            invtransform = glm::inverse(transform);
            return *this;
        }
        inline SceneNode& rotateX(const float angle_rad) { return rotate(angle_rad, glm::vec3(1, 0, 0)); }
        inline SceneNode& rotateY(const float angle_rad) { return rotate(angle_rad, glm::vec3(0, 1, 0)); }
        inline SceneNode& rotateZ(const float angle_rad) { return rotate(angle_rad, glm::vec3(0, 0, 1)); }
    };

    /**
     * @brief A class representing geometry in the scene
     * 
     */
    class GeometryNode: public SceneNode {
    private:
        /// @brief The type of geometry
        Geometry::Primitive m_primitive;

    public:
        /// @brief The object's material
        MaterialID materialID;

        /**
         * @brief Construct a new Geometry Node object
         * 
         * @param primitive The type of geometry
         * @param material The object's material
         */
        GeometryNode(Geometry::Primitive primitive, const Material& material)
            : SceneNode(MotionCurve{
                Curve<glm::vec3>::makeCurve<CurveType::CONSTANT>({glm::vec3(0.0)}),
                Curve<glm::mat4>::makeCurve<CurveType::CONSTANT>({glm::mat4(1.0)}),
                Curve<glm::vec3>::makeCurve<CurveType::CONSTANT>({glm::vec3(1.0)}),
            })
            , m_primitive(primitive)
            , materialID(material.id)
        {}

        /**
         * @brief Construct a new Geometry Node object
         * 
         * @param primitive The type of geometry
         * @param material The object's material
         */
        GeometryNode(Geometry::Primitive primitive, const Material& material, const MotionCurve& motionCurve_)
            : SceneNode(motionCurve_)
            , m_primitive(primitive)
            , materialID(material.id)
        {}

        /**
         * @brief Construct a new Geometry Node object
         * 
         * @param primitive The type of geometry
         * @param matID The object's material
         */
        GeometryNode(Geometry::Primitive primitive, MaterialID matID)
            : SceneNode(MotionCurve{
                Curve<glm::vec3>::makeCurve<CurveType::CONSTANT>({glm::vec3(0.0)}),
                Curve<glm::mat4>::makeCurve<CurveType::CONSTANT>({glm::mat4(1.0)}),
                Curve<glm::vec3>::makeCurve<CurveType::CONSTANT>({glm::vec3(1.0)}),
            })
            , m_primitive(primitive)
            , materialID(matID)
        {}

        /**
         * @brief Check whether a ray intersects with the geometry
         * 
         * @param hitInfo Output: information about how the ray intersects the geometry
         * @param r The ray to check
         * @param tRange The valid range for multiples of the ray's direction vector
         * @return true if the ray intersects with the geometry
         */
        __host__ __device__ bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const;
    };
}

#endif