#include "SceneNode.cuh"

using namespace PPCast;

__host__ __device__ bool GeometryNode::getIntersection(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) const {
    // Transform ray into local coordinates
    glm::mat4 toLocal = invtransform;
    const glm::mat4 timeTransform = motionCurve.at(r.time());
    const float determinant = glm::determinant(timeTransform);
    if (glm::abs(determinant) > 1e-8) {
        toLocal = timeTransform * toLocal;
    }

    Ray localRay(toLocal * r.origin(), toLocal * r.direction(), 0);

    // Compute intersection
    if (!Geometry::intersect(hitInfo, m_primitive, localRay, tRange)) return false;

    // Record HitInfo in original coordinates
    hitInfo.hitPoint   = r.at(hitInfo.t);
    const glm::vec3 outwardNormal3 = glm::normalize(glm::vec3(glm::transpose(toLocal) * hitInfo.normal));
    const glm::vec4 outwardNormal4 = glm::vec4(outwardNormal3, 0);
    hitInfo.hitOutside = glm::dot(r.direction(), outwardNormal4) < 0.f;
    hitInfo.normal     = hitInfo.hitOutside ? outwardNormal4 : -outwardNormal4;
    hitInfo.materialID = materialID;

    return true;
}