#include "SceneNode.h"

using namespace PPCast;

bool GeometryNode::getIntersection(HitInfo& hitInfo, const Ray& r) const {
    // Transform ray into local coordinates
    Ray localRay(invtransform * r.origin(), invtransform * r.direction(), r.interval());

    // Compute intersection
    if (!Geometry::intersect(hitInfo, m_primitive, localRay)) return false;

    // Record HitInfo in original coordinates
    hitInfo.hitPoint   = r.at(hitInfo.t);
    const glm::vec4 outwardNormal = glm::normalize(hitInfo.hitPoint - transform * glm::vec4(0, 0, 0, 1));
    hitInfo.hitOutside = glm::dot(r.direction(), outwardNormal) < 0.f;
    hitInfo.normal     = hitInfo.hitOutside ? outwardNormal : -outwardNormal;
    hitInfo.material   = material;
    assert(hitInfo.material != nullptr);

    return true;
}