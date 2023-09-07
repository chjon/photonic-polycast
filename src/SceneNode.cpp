#include "SceneNode.h"

using namespace PPCast;

bool GeometryNode::getIntersection(HitInfo& hitInfo, const Ray& r) const {
    // Transform ray into local coordinates
    Ray localRay(invtransform * r.origin(), invtransform * r.direction(), r.interval());

    // Compute intersection
    if (!Geometry::intersect(hitInfo, m_primitive, localRay)) return false;

    // Record HitInfo in original coordinates
    hitInfo.hitPoint   = r.at(hitInfo.t);
    const glm::vec3 outwardNormal3 = glm::normalize(glm::vec3(glm::transpose(invtransform) * hitInfo.normal));
    const glm::vec4 outwardNormal4 = glm::vec4(outwardNormal3, 0);
    hitInfo.hitOutside = glm::dot(r.direction(), outwardNormal4) < 0.f;
    hitInfo.normal     = hitInfo.hitOutside ? outwardNormal4 : -outwardNormal4;
    hitInfo.material   = material;
    assert(hitInfo.material != nullptr);

    return true;
}