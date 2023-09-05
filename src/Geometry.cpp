#include "Geometry.h"

using namespace PPCast;

bool Geometry::intersectSphere(float& t, glm::vec4& normal, const Ray& r) {
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
    t      = root;
    normal = r.at(root);

    return true;
}

static inline bool intersectCubePlane(float& t, const glm::vec3& n, const Ray& r) {
    const glm::vec3 o(r.origin());
    const glm::vec3 d(r.direction());
    const glm::vec3 u = n.yzx();
    const glm::vec3 v = n.zxy();
    const glm::vec3 coeffs = glm::inverse(glm::mat3(u, v, d)) * (o - n);
    const glm::vec3 abscoeffs = glm::abs(coeffs);
    if (abscoeffs.x > 1 || abscoeffs.y > 1 || !r.interval().contains(-coeffs.z)) return false;
    t = -coeffs.z;
    return true;
}

bool Geometry::intersectCube(float& t, glm::vec4& normal, const Ray& r) {
    const std::vector<glm::vec3> normals = {
        {+1, 0, 0}, {0, +1, 0}, {0, 0, +1},
        {-1, 0, 0}, {0, -1, 0}, {0, 0, -1},
    };

    bool intersected = false;
    for (const glm::vec3& n : normals) {
        float tmpT = std::numeric_limits<float>::infinity();
        if (!intersectCubePlane(tmpT, n, r)) continue;
        if (tmpT < t) {
            normal = glm::vec4(n, 0);
            t      = tmpT;
        }
        intersected = true;
    }

    return intersected;
}

bool Geometry::intersect(float& t, glm::vec4& normal, Primitive p, const Ray& r) {
    // Check for intersection
    switch (p) {
        case Primitive::Sphere: return intersectSphere(t, normal, r);
        case Primitive::Cube  : return intersectCube  (t, normal, r);
        default: return false;
    }
}