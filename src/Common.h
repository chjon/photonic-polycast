#ifndef PPCAST_COMMON_H
#define PPCAST_COMMON_H

#define GLM_FORCE_SWIZZLE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Utils
float randomFloat();
float randomFloat(float min, float max);

glm::vec4 randomOnHemisphere(const glm::vec4& normal);

// Printing GLM vectors and matrices
std::ostream& operator<<(std::ostream &os, const glm::vec2& v);
std::ostream& operator<<(std::ostream &os, const glm::vec3& v);
std::ostream& operator<<(std::ostream &os, const glm::vec4& v);
std::ostream& operator<<(std::ostream &os, const glm::mat3& v);
std::ostream& operator<<(std::ostream &os, const glm::mat4& v);

template <uint8_t L> glm::vec<L, float, glm::packed_highp> randomFloatVector() {
    glm::vec<L, float, glm::packed_highp> v;
    for (uint8_t i = 0; i < L; ++i) v[i] = randomFloat(-1.f, 1.f);
    return v;
}

template <uint8_t L>
glm::vec<L, float, glm::packed_highp> randomInUnitSphere() {
    while (true) {
        const glm::vec<L, float, glm::packed_highp> candidate = randomFloatVector<L>();
        if (glm::dot(candidate, candidate) < 1.f) return candidate;
    }
}

template <uint8_t L>
glm::vec<L, float, glm::packed_highp> randomUnitVector() { return glm::normalize(randomInUnitSphere<L>()); }

#endif