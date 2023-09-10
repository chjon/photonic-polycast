#ifndef PPCAST_SCENENODE_H
#define PPCAST_SCENENODE_H

#include "Common.h"
#include "Geometry.h"
#include "Material.h"
#include "Renderable.h"

namespace PPCast {
    class SceneNode {
    protected:
        glm::mat4x4 transform;
        glm::mat4x4 invtransform;

    public:
        SceneNode() : transform(1.f), invtransform(1.f) {}

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

    class GeometryNode: public SceneNode, public Renderable {
    private:
        Geometry::Primitive m_primitive;

    public:
        MaterialID materialID;

        GeometryNode(Geometry::Primitive primitive, const Material& material)
            : SceneNode()
            , m_primitive(primitive)
            , materialID(material.id)
        {}

        GeometryNode(Geometry::Primitive primitive, MaterialID matID)
            : SceneNode()
            , m_primitive(primitive)
            , materialID(matID)
        {}

        __host__ __device__ virtual bool getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const override;
    };
}

#endif