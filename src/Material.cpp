#include "Material.h"

using namespace PPCast;

bool MaterialNormal::scatter(
    glm::vec4&,
    glm::vec3& attenuation,
    const glm::vec4&,
    const HitInfo& hitInfo
) const {
    attenuation = 0.5f * (glm::normalize(hitInfo.normal.xyz()) + glm::vec3(1));
    return false;
}

static inline glm::vec4 reflect(const glm::vec4& directionIn, const glm::vec4& normal) {
    return directionIn - 2.f * glm::dot(directionIn, normal) * glm::normalize(normal);
}

static inline glm::vec4 refract(const glm::vec4& uv, const glm::vec4& n, const float r) {
    const float cos_theta    = glm::min(glm::dot(-uv, n), 1.f);
    glm::vec4 r_out_perp     = r * (uv + cos_theta*n);
    glm::vec4 r_out_parallel = -glm::sqrt(glm::abs(1.f - glm::dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

bool MaterialReflDir::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const HitInfo& hitInfo
) const {
    directionOut = reflect(directionIn, hitInfo.normal);
    attenuation = 0.5f * (glm::normalize(directionOut.xyz()) + glm::vec3(1));
    return false;
}

bool MaterialRefrDir::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const HitInfo& hitInfo
) const {
    const glm::vec4 unitDirIn = glm::normalize(directionIn);
    const glm::vec4 unitNormal = glm::normalize(hitInfo.normal);
    const float c = glm::min(glm::dot(-unitDirIn, unitNormal), 1.f);
    const float s = glm::sqrt(1.f - c*c);

    // Decide whether to reflect or refract
    const float refractionRatio = (hitInfo.hitOutside) ? (1.f / m_refractiveIndex) : m_refractiveIndex;    
    if (refractionRatio * s > 1.0) directionOut = reflect(unitDirIn, unitNormal);
    else                           directionOut = refract(unitDirIn, unitNormal, refractionRatio);
    attenuation = 0.5f * (glm::normalize(directionOut.xyz()) + glm::vec3(1));
    return false;
}

bool MaterialDiffuse::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4&,
    const HitInfo& hitInfo
) const {
    directionOut = randomOnHemisphere(hitInfo.normal);
    attenuation = m_attenuation;
    return true;
}

bool MaterialLambertian::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4&,
    const HitInfo& hitInfo
) const {
    glm::vec4 unitNormal = glm::normalize(hitInfo.normal);
    directionOut = unitNormal + glm::vec4(randomUnitVector<3>(), 0);
    if (glm::dot(directionOut, directionOut) < 1e-8) directionOut = unitNormal;
    attenuation = m_attenuation;
    return true;
}

bool MaterialMetal::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const HitInfo& hitInfo
) const {
    const glm::vec4 reflected = reflect(directionIn, hitInfo.normal);
    directionOut = reflected + glm::vec4(m_fuzzFactor * randomUnitVector<3>(), 0);
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
    const HitInfo& hitInfo
) const {
    attenuation = m_attenuation;

    const glm::vec4 unitDirIn = glm::normalize(directionIn);
    const glm::vec4 unitNormal = glm::normalize(hitInfo.normal);
    const float c = glm::min(glm::dot(-unitDirIn, unitNormal), 1.f);
    const float s = glm::sqrt(1.f - c*c);

    // Decide whether to reflect or refract
    const float refractionRatio = (hitInfo.hitOutside) ? (1.f / m_refractiveIndex) : m_refractiveIndex;    
    const bool shouldReflect = (
        refractionRatio * s > 1.0 ||                    // Critical angle -- Snell's Law
        randomFloat() < reflectance(c, refractionRatio) // Probabilistic reflection
    );
    if (shouldReflect) directionOut = reflect(unitDirIn, unitNormal);
    else               directionOut = refract(unitDirIn, unitNormal, refractionRatio);
    return true;
}