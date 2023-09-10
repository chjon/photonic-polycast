#include "Material.h"

using namespace PPCast;

static inline glm::vec4 reflect(const glm::vec4& directionIn, const glm::vec4& normal) {
    return directionIn - 2.f * glm::dot(directionIn, normal) * glm::normalize(normal);
}

static inline glm::vec4 refract(const glm::vec4& uv, const glm::vec4& n, const float r) {
    const float cos_theta    = glm::min(glm::dot(-uv, n), 1.f);
    glm::vec4 r_out_perp     = r * (uv + cos_theta*n);
    glm::vec4 r_out_parallel = -glm::sqrt(glm::abs(1.f - glm::dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

// Schlick's approximation for probability of reflection
static inline float reflectance(float cosine, float refractiveIndex) {
    float r0 = (1.f - refractiveIndex) / (1.f + refractiveIndex);
    r0 = r0 * r0;
    return r0 + (1.f - r0) * static_cast<float>(glm::pow((1.f - glm::abs(cosine)), 5.f));
}

static inline glm::vec3 dir2Colour(const glm::vec4& dir) {
    return 0.5f * (glm::normalize(dir.xyz()) + glm::vec3(1));
}

static inline bool scatterNormalDir(
    glm::vec4&,
    glm::vec3& attenuation,
    const glm::vec4&,
    const HitInfo& hitInfo,
    const MaterialData&
) {
    attenuation = dir2Colour(hitInfo.normal);
    return false;
}

static inline bool scatterReflectDir(
    glm::vec4&,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const HitInfo& hitInfo,
    const MaterialData&
) {
    attenuation = dir2Colour(reflect(directionIn, hitInfo.normal));
    return false;
}

static inline bool scatterRefractDir(
    glm::vec4&,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const HitInfo& hitInfo,
    const MaterialData& data
) {
    const glm::vec4 unitDirIn = glm::normalize(directionIn);
    const glm::vec4 unitNormal = glm::normalize(hitInfo.normal);
    const float c = glm::min(glm::dot(-unitDirIn, unitNormal), 1.f);
    const float s = glm::sqrt(1.f - c*c);

    // Decide whether to reflect or refract
    const float ir = data.indexOfRefraction;
    const float refractionRatio = (hitInfo.hitOutside) ? (1.f / ir) : ir;    
    if (refractionRatio * s > 1.0) attenuation = dir2Colour(reflect(unitDirIn, unitNormal));
    else                           attenuation = dir2Colour(refract(unitDirIn, unitNormal, refractionRatio));
    return false;
}

static inline bool scatterDiffuse(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4&,
    const HitInfo& hitInfo,
    const MaterialData& data
) {
    directionOut = randomOnHemisphere(hitInfo.normal);
    attenuation  = data.attenuation;
    return true;
}

static inline bool scatterLambertian(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4&,
    const HitInfo& hitInfo,
    const MaterialData& data
) {
    const glm::vec4 unitNormal = glm::normalize(hitInfo.normal);
    directionOut = unitNormal + glm::vec4(randomUnitVector<3>(), 0);
    if (glm::dot(directionOut, directionOut) < 1e-8) directionOut = unitNormal;
    attenuation = data.attenuation;
    return true;
}

static inline bool scatterReflective(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const HitInfo& hitInfo,
    const MaterialData& data
) {
    const glm::vec4 reflected = reflect(directionIn, hitInfo.normal);
    directionOut = reflected + glm::vec4(data.fuzzFactor * randomUnitVector<3>(), 0);
    attenuation = data.attenuation;
    return true;
}

static inline bool scatterRefractive(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const HitInfo& hitInfo,
    const MaterialData& data
) {
    attenuation = data.attenuation;

    const glm::vec4 unitDirIn = glm::normalize(directionIn);
    const glm::vec4 unitNormal = glm::normalize(hitInfo.normal);
    const float c = glm::min(glm::dot(-unitDirIn, unitNormal), 1.f);
    const float s = glm::sqrt(1.f - c*c);

    // Decide whether to reflect or refract
    const float ir = data.indexOfRefraction;
    const float refractionRatio = (hitInfo.hitOutside) ? (1.f / ir) : ir;    
    const bool shouldReflect = (
        refractionRatio * s > 1.0 ||                    // Critical angle -- Snell's Law
        randomFloat() < reflectance(c, refractionRatio) // Probabilistic reflection
    );
    if (shouldReflect) directionOut = reflect(unitDirIn, unitNormal);
    else               directionOut = refract(unitDirIn, unitNormal, refractionRatio);
    return true;
}

bool Material::scatter(
    glm::vec4& directionOut,
    glm::vec3& attenuation,
    const glm::vec4& directionIn,
    const HitInfo& hitInfo
) const {
    switch (type) {
        case MaterialType::NormalDir:  return scatterNormalDir (directionOut, attenuation, directionIn, hitInfo, data);
        case MaterialType::ReflectDir: return scatterReflectDir(directionOut, attenuation, directionIn, hitInfo, data);
        case MaterialType::RefractDir: return scatterRefractDir(directionOut, attenuation, directionIn, hitInfo, data);
        case MaterialType::Diffuse:    return scatterDiffuse   (directionOut, attenuation, directionIn, hitInfo, data);
        case MaterialType::Lambertian: return scatterLambertian(directionOut, attenuation, directionIn, hitInfo, data);
        case MaterialType::Reflective: return scatterReflective(directionOut, attenuation, directionIn, hitInfo, data);
        case MaterialType::Refractive: return scatterRefractive(directionOut, attenuation, directionIn, hitInfo, data);
        default: attenuation = glm::vec3(0.f); return false;
    }
}