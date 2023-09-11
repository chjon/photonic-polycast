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

// Printing GLM vectors and matrices
std::ostream& operator<<(std::ostream &os, const glm::vec2& v);
std::ostream& operator<<(std::ostream &os, const glm::vec3& v);
std::ostream& operator<<(std::ostream &os, const glm::vec4& v);
std::ostream& operator<<(std::ostream &os, const glm::mat3& v);
std::ostream& operator<<(std::ostream &os, const glm::mat4& v);

#endif