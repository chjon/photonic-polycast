#include "Common.h"

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