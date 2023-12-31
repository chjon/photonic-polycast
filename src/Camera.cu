#include "Camera.cuh"
#include "Image.h"
#include "Material.cuh"
#include "Options.h"
#include "Random.cuh"
#include "World.cuh"

using namespace PPCast;

static BoolOption  opt_usegpu    ("usegpu"     , "whether to use the GPU for rendering" , false);
static FloatOption opt_vfov      ("vfov"       , "the camera's vertical FOV in degrees" , 90.0f);
static FloatOption opt_dofAngle  ("dof-angle"  , "the camera's defocus angle in degrees",  0.0f);
static FloatOption opt_focalDist ("focal-dist" , "the camera's focal distance"          ,  1.0f);
static UIntOption  opt_raysPerPx ("rays-per-px", "number of rays to cast per pixel"     ,    10);
static UIntOption  opt_maxBounces("max-bounces", "maximum number of ray bounces"        ,     5);
static UIntOption  opt_seed      ("seed"     , "random seed"                            , 0xDECAFBAD);

Camera::Camera()
    : vfov      (*opt_vfov      )
    , dofAngle  (*opt_dofAngle  )
    , focalDist (*opt_focalDist )
    , raysPerPx (*opt_raysPerPx )
    , maxBounces(*opt_maxBounces)
    , seed      (*opt_seed      )
    , useGPU    (*opt_usegpu    )
{}

bool Camera::renderImageCPU(Image& image, const World& scene) const {
    std::mt19937 randomGenerator(seed);
    RandomState randomState(&randomGenerator);
    for (uint32_t y = 0; y < height; ++y) {
        std::clog << "\rRendering scanlines: " << (y + 1) << " / " << height << " " << std::flush;
        for (uint32_t x = 0; x < width; ++x) {
            image.get(x, y) = renderPixel(x, y, scene, randomState);
        }
    }
    std::clog << "\rRendering completed: " << height << " / " << height << std::flush << std::endl;

    return true;
}

__host__ __device__ Ray Camera::generateRay(uint32_t x, uint32_t y, RandomState& randomState) const {
    // Compute origin point in viewspace
    glm::vec3 rayOrigin = {0, 0, 0}; 

    // Perturb the origin point
    rayOrigin += m_defocusRadius * glm::vec3(randomInUnitSphere<2>(randomState), 0);

    // Compute the pixel sample position in the viewspace focal plane
    glm::vec3 pixelSample =
        m_pixel_topLeft +
        (static_cast<float>(x) * m_pixel_dx) +
        (static_cast<float>(y) * m_pixel_dy);
    
    // Perturb the pixel sample position
    pixelSample += (m_pixel_dx + m_pixel_dy) * glm::vec3(randomFloatVector<2>(randomState) - 0.5f * glm::vec2(1), 0);

    // Compute the pixel sample position in the worldspace focal plane
    glm::vec4 worldspacePixelSample = m_v2w * glm::vec4(pixelSample, 1);

    // Sample a random ray emission time
    const float sampleTime = randomFloat(randomState);

    // Return ray in worldspace
    const glm::vec4 worldspaceRayOrigin = m_v2w * glm::vec4(rayOrigin, 1);
    const glm::vec4 worldspaceRayDir    = glm::normalize(worldspacePixelSample - worldspaceRayOrigin);
    return Ray(worldspaceRayOrigin, worldspaceRayDir, sampleTime);
}

__host__ __device__ static glm::vec3 getSkyboxColour(const glm::vec4& direction) {
    constexpr glm::vec3 skyTopColour    = glm::vec3(0.5, 0.7, 1.0);
    constexpr glm::vec3 skyBottomColour = glm::vec3(1.0, 1.0, 1.0);
    const float a = 0.5f * (glm::normalize(direction).y + 1.f);
    return (1.f - a) * skyBottomColour + a * skyTopColour;
}

__host__ __device__ glm::vec3 Camera::raycast(
    const Ray& ray, Interval<float>&& tRange,
    const World& world, RandomState& randomState
) const {
    Ray currentRay = ray;
    glm::vec3 totalAttenuation(1);

    uint32_t maxDepth = maxBounces;
    while (maxDepth--) {
        // Check if ray intersects with geometry
        HitInfo hitInfo;
        const bool hit = world.getIntersection(hitInfo, currentRay, tRange);

        // Return the sky colour if there is no intersection
        if (!hit) return totalAttenuation * getSkyboxColour(currentRay.direction());
        
        // Generate a reflected or transmitted ray
        glm::vec4 scatterDirection;
        glm::vec3 attenuation;
        assert(static_cast<uint32_t>(hitInfo.materialID) != UINT32_MAX);
        const Material& material = world.materials[static_cast<uint32_t>(hitInfo.materialID)];
        const bool generatedRay = material.scatter(
            scatterDirection,
            attenuation,
            currentRay.direction(),
            hitInfo,
            randomState
        );
        
        // Return the material colour if there is no generated ray
        if (!generatedRay) return totalAttenuation * attenuation;

        // Cast the generated ray
        tRange.lower = 1e-3f;
        currentRay = Ray(hitInfo.hitPoint, glm::normalize(scatterDirection), ray.time());
        totalAttenuation = totalAttenuation * attenuation;
    }

    return glm::vec3(0);
}

void Camera::initialize(uint32_t w, uint32_t h) {
    width  = w;
    height = h;

    // Compute viewport params
    const float     aspectRatio      = static_cast<float>(w) / static_cast<float>(h);
    const float     viewport_h       = 2 * glm::tan(glm::radians(0.5f * vfov));
    const float     viewport_w       = aspectRatio * viewport_h;
    const glm::vec3 viewport_x       = focalDist * glm::vec3(viewport_w, 0, 0);
    const glm::vec3 viewport_y       = focalDist * glm::vec3(0, -viewport_h, 0);
    const glm::vec3 viewport_topLeft = focalDist * glm::vec3(0, 0, -1) - 0.5f * (viewport_x + viewport_y);

    m_pixel_dx      = viewport_x / static_cast<float>(width);
    m_pixel_dy      = viewport_y / static_cast<float>(height);
    m_pixel_topLeft = viewport_topLeft + 0.5f * (m_pixel_dx + m_pixel_dy);
    m_v2w           = glm::inverse(glm::lookAt(lookfrom, lookat, up));
    m_defocusRadius = focalDist * glm::tan(glm::radians(0.5f * dofAngle));
}

__host__ __device__ glm::vec3 Camera::renderPixel(
    uint32_t x, uint32_t y,
    const World& world,
    RandomState& randomState
) const {
    // Perform raytracing
    glm::vec3 total(0);
    Interval<float> tRange(0.f, INFINITY, true, false);
    for (uint32_t i = 0; i < raysPerPx; ++i) {
        const Ray ray = generateRay(x, y, randomState);
        total += raycast(ray, std::move(tRange), world, randomState);
    }
    return total / static_cast<float>(raysPerPx);
}