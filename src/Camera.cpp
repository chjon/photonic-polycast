#include "Camera.h"
#include "Material.h"
#include "Options.h"
#include "Ray.h"
#include "World.h"

using namespace PPCast;

static FloatOption opt_cam_fovy  ("cam-fovy"   , "the camera's y-axis FOV in degrees",  90.0f);
static UIntOption  opt_raysPerPx ("rays-per-px", "number of rays to cast per pixel"  ,     10);
static UIntOption  opt_maxBounces("max-bounces", "maximum number of ray bounces"     ,      5);

Camera::Camera()
    : fovy      (*opt_cam_fovy  )
    , raysPerPx (*opt_raysPerPx )
    , maxBounces(*opt_maxBounces)
{}

static png::rgb_pixel vec2Pixel(const glm::vec3& v) {
    const glm::vec3 colour = 255.99f * glm::sqrt(v);
    return png::rgb_pixel(
        static_cast<unsigned char>(colour.r),
        static_cast<unsigned char>(colour.g),
        static_cast<unsigned char>(colour.b)
    );
}

static glm::vec3 getSkyboxColour(const glm::vec4& direction) {
    constexpr glm::vec3 skyTopColour    = glm::vec3(0.5, 0.7, 1.0);
    constexpr glm::vec3 skyBottomColour = glm::vec3(1.0, 1.0, 1.0);
    const float a = 0.5f * (glm::normalize(direction).y + 1.f);
    return (1.f - a) * skyBottomColour + a * skyTopColour;
}

glm::vec3 Camera::raycast(const Ray& ray, Interval<float>&& tRange, const World& world, unsigned int maxDepth) {
    Ray currentRay = ray;
    glm::vec3 totalAttenuation(1);

    while (maxDepth--) {
        // Check if ray intersects with geometry
        HitInfo hitInfo;
        const bool hit = world.getIntersection(hitInfo, currentRay, tRange);

        // Return the sky colour if there is no intersection
        if (!hit) return totalAttenuation * getSkyboxColour(currentRay.direction());
        
        // Generate a reflected or transmitted ray
        glm::vec4 scatterDirection;
        glm::vec3 attenuation;
        const bool generatedRay = hitInfo.material->scatter(scatterDirection, attenuation, currentRay.direction(), hitInfo);
        
        // Return the material colour if there is no generated ray
        if (!generatedRay) return totalAttenuation * attenuation;

        // Cast the generated ray
        tRange.min = 1e-3f;
        currentRay = Ray(hitInfo.hitPoint, glm::normalize(scatterDirection));
        totalAttenuation = totalAttenuation * attenuation;
    }

    return glm::vec3(0);
}

Ray Camera::generateRay(uint32_t x, uint32_t y) const {
    // Compute direction vector in viewspace
    glm::vec3 rayDirection =
        m_pixel_topLeft +
        (static_cast<float>(x) * m_pixel_dx) +
        (static_cast<float>(y) * m_pixel_dy);
    
    // Perturb the direction vector
    const float randomX = randomFloat() - 0.5f;
    const float randomY = randomFloat() - 0.5f;
    rayDirection += randomX * m_pixel_dx + randomY * m_pixel_dy;

    // Return ray in worldspace
    const glm::vec4 worldspaceRayOrigin = m_v2w * glm::vec4(0, 0, 0, 1);
    const glm::vec4 worldspaceRayDir    = glm::normalize(m_v2w * glm::vec4(rayDirection, 0));
    return Ray(worldspaceRayOrigin, worldspaceRayDir);
}

void Camera::initialize(uint32_t w, uint32_t h) {
    width  = w;
    height = h;

    // Compute viewport params
    const float     aspectRatio      = static_cast<float>(w) / static_cast<float>(h);
    const float     focalLength      = 0.5f / glm::tan(glm::radians(0.5f * fovy));
    const glm::vec3 viewport_x       = glm::vec3(aspectRatio, 0, 0);
    const glm::vec3 viewport_y       = glm::vec3(0, -1, 0);
    const glm::vec3 viewport_topLeft = -glm::vec3(0, 0, focalLength) - 0.5f * (viewport_x + viewport_y);

    m_pixel_dx      = viewport_x / static_cast<float>(width);
    m_pixel_dy      = viewport_y / static_cast<float>(height);
    m_pixel_topLeft = viewport_topLeft + 0.5f * (m_pixel_dx + m_pixel_dy);
    m_v2w           = glm::inverse(glm::lookAt(pos, centre, up));
}

glm::vec3 Camera::renderPixel(uint32_t x, uint32_t y, const World& scene) const {
    // Perform raytracing
    glm::vec3 total(0);
    Interval<float> tRange(0.f, std::numeric_limits<float>::infinity(), true, false);
    for (uint32_t i = 0; i < raysPerPx; ++i) {
        const Ray ray = generateRay(x, y);
        total += raycast(ray, std::move(tRange), scene, maxBounces);
    }
    return total / static_cast<float>(raysPerPx);
}

png::image<png::rgb_pixel> Camera::renderImage(const World& scene) const {
    png::image<png::rgb_pixel> image(width, height);
    for (png::uint_32 y = 0; y < height; ++y) {
        std::clog << "\rRendering scanlines: " << (y + 1) << " / " << height << " " << std::flush;
        for (png::uint_32 x = 0; x < width; ++x) {
            image[y][x] = vec2Pixel(renderPixel(x, y, scene));
        }
    }
    std::clog << "\rRendering completed: " << height << " / " << height << std::flush << std::endl;

    return image;
}