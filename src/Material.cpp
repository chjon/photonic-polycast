#include "Material.h"

using namespace PPCast;

bool MaterialNormal::scatter(
    glm::vec4&,
    glm::vec3& attenuation,
    const glm::vec4&,
    const glm::vec4& normal
) const {
    attenuation = 0.5f * (glm::normalize(normal.xyz()) + glm::vec3(1));
    return false;
}

bool MaterialDiffuse::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4&,
    const glm::vec4& normal
) const {
    directionOut = randomOnHemisphere(normal);
    attenuation = m_attenuation;
    return true;
}

bool MaterialLambertian::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4&,
    const glm::vec4& normal
) const {
    directionOut = glm::normalize(normal) + glm::vec4(randomUnitVector(), 0);
    if (glm::dot(directionOut, directionOut) < 1e-8) directionOut = glm::normalize(normal);
    attenuation = m_attenuation;
    return true;
}

bool MaterialMetal::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const glm::vec4& normal
) const {
    const glm::vec4 reflected = directionIn - 2.f * glm::dot(directionIn, normal) * glm::normalize(normal);
    directionOut = reflected + glm::vec4(m_fuzzFactor * randomUnitVector(), 0);
    attenuation = m_attenuation;
    return true;
}