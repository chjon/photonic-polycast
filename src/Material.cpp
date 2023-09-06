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

static inline glm::vec4 reflect(const glm::vec4& directionIn, const glm::vec4& normal) {
    return directionIn - 2.f * glm::dot(directionIn, normal) * glm::normalize(normal);
}

static inline glm::vec4 refract(const glm::vec4& dirIn, const glm::vec4& n, const float r) {
    const float c = glm::dot(dirIn, n);
    return (r * c - glm::sqrt(1.f - r * r * (1.f - c * c))) * n - r * dirIn;
}

bool MaterialMetal::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const glm::vec4& normal
) const {
    const glm::vec4 reflected = reflect(directionIn, normal);
    directionOut = reflected + glm::vec4(m_fuzzFactor * randomUnitVector(), 0);
    attenuation = m_attenuation;
    return true;
}

// Schlick's approximation for probability of reflection
static inline float reflectance(float cosine, float refractiveIndex) {
    float r0 = (1.f - refractiveIndex) / (1.f + refractiveIndex);
    r0 = r0 * r0;
    return r0 + (1.f - r0) * static_cast<float>(glm::pow((1.f - glm::abs(cosine)), 5.f));
}

bool MaterialRefractive::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const glm::vec4& normal
) const {
    attenuation = m_attenuation;

    const glm::vec4 unitDirIn = glm::normalize(directionIn);
    const glm::vec4 unitNormal = glm::normalize(normal);
    const float c = glm::min(glm::dot(-unitDirIn, unitNormal), 1.f);
    const float s = glm::sqrt(1.f - c*c);

    // Decide whether to reflect or refract
    const float refractionRatio = (glm::dot(unitDirIn, unitNormal) < 0) ? (1.f / m_refractiveIndex) : m_refractiveIndex;    
    const bool shouldReflect = (
        refractionRatio * s > 1.0 ||                    // Critical angle -- Snell's Law
        randomFloat() < reflectance(c, refractionRatio) // Probabilistic reflection
    );
    if (shouldReflect) directionOut = reflect(unitDirIn, unitNormal);
    else               directionOut = refract(unitDirIn,-unitNormal, refractionRatio);
    return true;
}