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
#include <iostream>
#include <limits>
#include <string>
#include <vector>

std::ostream& operator<<(std::ostream &os, const glm::vec2& v);
std::ostream& operator<<(std::ostream &os, const glm::vec3& v);
std::ostream& operator<<(std::ostream &os, const glm::vec4& v);
std::ostream& operator<<(std::ostream &os, const glm::mat3& v);
std::ostream& operator<<(std::ostream &os, const glm::mat4& v);

#endif