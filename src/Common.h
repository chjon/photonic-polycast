#ifndef PPCAST_COMMON_H
#define PPCAST_COMMON_H

#include <cuda.h>

#define GLM_FORCE_CUDA
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Printing GLM vectors
template <glm::length_t L, typename T, glm::qualifier Q>
std::ostream& operator<<(std::ostream &os, const glm::vec<L, T, Q>& v) {
    os << '(' << v[0];
    for (glm::length_t i = 1; i < L; ++i)
        os << ',' << v[i];
    return os << ')';
}

// Printing GLM matrices
template <glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
std::ostream& operator<<(std::ostream &os, const glm::mat<C, R, T, Q>& m) {
    os << '(' << m[0];
    for (glm::length_t i = 1; i < C; ++i)
        os << ',' << m[i];
    return os << ')';
}

#endif