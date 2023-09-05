#include "SceneNode.h"

using namespace PPCast;

static bool intersectSphere(HitInfo& hitInfo, const Ray& r) {
    // Solving a*t^2 + 2*b*t + c = 0
    const glm::vec3 o = glm::vec3(r.origin());
    const glm::vec3 d = glm::vec3(r.direction());
    const float     a = glm::dot(d, d);
    const float     b = glm::dot(o, d);
    const float     c = glm::dot(o, o) - 1;

    // Find nearest root
    const float discriminant = b * b - a * c;
    if (discriminant < 0) return false;
    const float sqrt_discriminant = glm::sqrt(discriminant);
    float root = (-b - sqrt_discriminant) / a;
    if (!r.interval().contains(root)) {
        root = (-b + sqrt_discriminant) / a;
        if (!r.interval().contains(root)) return false;
    }

    // Update hit info
    hitInfo.t        = root;
    hitInfo.hitPoint = r.at(root);
    hitInfo.normal   = hitInfo.hitPoint;

    return true;
}

bool GeometryNode::getIntersection(HitInfo& hitInfo, const Ray& r) const {
    // Transform ray into local coordinates
    Ray localRay(invtransform * r.origin(), invtransform * r.direction(), r.interval());

    // Check for intersection
    switch (m_geometry) {
        case Geometry::Sphere:
            if (!intersectSphere(hitInfo, localRay)) return false;
            break; 
        default: return false;
    }
    
    // Transform HitInfo back into original coordinates
    hitInfo.hitPoint = transform                    * hitInfo.hitPoint;
    hitInfo.normal   = glm::transpose(invtransform) * hitInfo.normal;

    return true;
}