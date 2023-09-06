#include "SceneNode.h"

using namespace PPCast;

bool GeometryNode::getIntersection(HitInfo& hitInfo, const Ray& r) const {
    // Transform ray into local coordinates
    Ray localRay(invtransform * r.origin(), invtransform * r.direction(), r.interval());

    // Compute intersection
    if (!Geometry::intersect(hitInfo.t, hitInfo.normal, m_primitive, localRay)) return false;

    // Record HitInfo in original coordinates
    hitInfo.hitPoint = r.at(hitInfo.t);
    hitInfo.normal   = glm::normalize(hitInfo.hitPoint - transform * glm::vec4(0, 0, 0, 1));
    hitInfo.material = material;

    return true;
}