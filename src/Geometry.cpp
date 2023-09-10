#include "Geometry.h"

using namespace PPCast;

__host__ __device__ bool Geometry::intersectSphere(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) {
    // Solving a*t^2 + 2*b*t + c = 0
    const glm::vec3 o = glm::vec3(r.origin());
    const glm::vec3 d = glm::vec3(r.direction());
    const float     a = glm::dot(d, d);
    const float     b = glm::dot(o, d);
    const float     c = glm::dot(o, o) - 1;

    // Check whether a root exists
    const float discriminant = b * b - a * c;
    if (discriminant < 0) return false;

    // Find the nearest root
    const float sqrt_discriminant = glm::sqrt(discriminant);
    float root;
    if (!tRange.contains(root = ((-b - sqrt_discriminant) / a)) &&
        !tRange.contains(root = ((-b + sqrt_discriminant) / a))
    ) return false;

    // Update hit info
    hitInfo.t      = root;
    hitInfo.normal = glm::vec4(r.at(root).xyz(), 0);
    return true;
}

__host__ __device__ static inline bool intersectCubePlane(float& t, const glm::vec3& n, const Ray& r, const Interval<float>& tRange) {
    const glm::vec3 o(r.origin());
    const glm::vec3 d(r.direction());
    const glm::vec3 u = n.yzx();
    const glm::vec3 v = n.zxy();
    const glm::mat3 uvd = glm::mat3(u, v, d);
    const float determinant = glm::determinant(uvd);
    if (glm::abs(determinant) < 1e-8) return false;
    const glm::vec3 coeffs = glm::inverse(uvd) * (o - n);
    const glm::vec3 abscoeffs = glm::abs(coeffs);
    if (abscoeffs.x > 1 || abscoeffs.y > 1 || !tRange.contains(-coeffs.z)) return false;
    t = -coeffs.z;
    return true;
}

__host__ __device__ bool Geometry::intersectCube(HitInfo& hitInfo, const Ray& r, const Interval<float>& tRange) {
    const std::vector<glm::vec3> normals = {
        {+1, 0, 0}, {0, +1, 0}, {0, 0, +1},
        {-1, 0, 0}, {0, -1, 0}, {0, 0, -1},
    };

    bool intersected = false;
    for (const glm::vec3& n : normals) {
        float tmpT = std::numeric_limits<float>::infinity();
        if (!intersectCubePlane(tmpT, n, r, tRange)) continue;
        if (tmpT < hitInfo.t) {
            hitInfo.normal = glm::vec4(n, 0);
            hitInfo.t      = tmpT;
        }
        intersected = true;
    }

    return intersected;
}

bool Geometry::intersect(HitInfo& hitInfo, Primitive p, const Ray& r, const Interval<float>& tRange) {
    // Check for intersection
    switch (p) {
        case Primitive::Sphere: return intersectSphere(hitInfo, r, tRange);
        case Primitive::Cube  : return intersectCube  (hitInfo, r, tRange);
        default: return false;
    }
}