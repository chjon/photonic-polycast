#include "Common.h"

float randomFloat() {
    return static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.f);
}

float randomFloat(float min, float max) {
    return min + (max - min) * randomFloat();
}

glm::vec4 randomOnHemisphere(const glm::vec4& normal) {
    glm::vec4 v = glm::vec4(randomUnitVector<3>(), 0);
    return (glm::dot(v, normal) < 0) ? -v : v;
}

std::ostream& operator<<(std::ostream &os, const glm::vec2& v) {
    return os << '(' << v.x << ',' << v.y << ')';
}

std::ostream& operator<<(std::ostream &os, const glm::vec3& v) {
    return os << '(' << v.x << ',' << v.y << ',' << v.z << ')';
}

std::ostream& operator<<(std::ostream &os, const glm::vec4& v) {
    return os << '(' << v.x << ',' << v.y << ',' << v.z << ',' << v.w << ')';
}

std::ostream& operator<<(std::ostream &os, const glm::mat3& m) {
    return os << '(' << m[0] << ',' << m[1] << ',' << m[2] << ')';
}

std::ostream& operator<<(std::ostream &os, const glm::mat4& m) {
    return os << '(' << m[0] << ',' << m[1] << ',' << m[2] << ',' << m[3] << ')';
}