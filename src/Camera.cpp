#include "Camera.h"
#include "Material.h"
#include "Options.h"
#include "Ray.h"

using namespace PPCast;

static FloatOption opt_cam_fovy   ("cam-fovy"  , "the camera's y-axis FOV in degrees"                 ,  90.0f);
static FloatOption opt_cam_aspect ("cam-aspect", "the camera's aspect ratio (x / y)"                  ,   1.0f);
static UIntOption  opt_jitter     ("jitter"    , "number of rays to cast per pixel (0 for no jitter)" ,      0);
static UIntOption  opt_max_depth  ("max-depth" , "maximum recursion depth"                            ,      5);

Camera::Camera(const glm::vec3& pos, const glm::vec3& centre, const glm::vec3& up)
    : m_pos   (pos)
    , m_centre(centre)
    , m_up    (up)
    , m_fovy  (*opt_cam_fovy)
    , m_aspect(*opt_cam_aspect)
    , m_jitter(*opt_jitter)
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

static glm::vec3 raycast(const Ray& ray, Interval<float>&& tRange, const std::vector<GeometryNode>& scene, unsigned int maxDepth) {
    Ray currentRay = ray;
    glm::vec3 totalAttenuation(1);

    while (maxDepth--) {
        // Check if ray intersects with geometry
        HitInfo hitInfo;
        bool hit = false;
        for (const GeometryNode& node : scene) {
            HitInfo tmpHitInfo;
            if (node.getIntersection(tmpHitInfo, currentRay, tRange)) {
                hit = true;
                if (tmpHitInfo.t < hitInfo.t) hitInfo = tmpHitInfo;
            }
        }

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

png::image<png::rgb_pixel> Camera::render(const std::vector<GeometryNode>& scene, uint32_t width, uint32_t height) const {
    // Compute viewport params
    const double    focalLength      = 0.5 / glm::tan(glm::radians(0.5 * m_fovy));
    const glm::vec3 viewport_x       = glm::vec3(1 * m_aspect,  0, 0);
    const glm::vec3 viewport_y       = glm::vec3(0, -1, 0);
    const glm::vec3 pixel_dx         = viewport_x / static_cast<float>(width);
    const glm::vec3 pixel_dy         = viewport_y / static_cast<float>(height);
    const glm::vec3 viewport_topLeft = -glm::vec3(0, 0, focalLength) - 0.5f * (viewport_x + viewport_y);
    const glm::vec3 pixel_topLeft    = viewport_topLeft + 0.5f * (pixel_dx + pixel_dy);
    const glm::mat4 vinv             = glm::inverse(getView());

    // Actually render the image
    png::image<png::rgb_pixel> image(width, height);
    for (png::uint_32 y = 0; y < height; ++y) {
        std::clog << "\rRendering scanlines: " << (y + 1) << " / " << height << " " << std::flush;
        for (png::uint_32 x = 0; x < width; ++x) {
            const glm::vec3 curr_pixel = pixel_topLeft + (static_cast<float>(x) * pixel_dx) + (static_cast<float>(y) * pixel_dy);
            glm::vec3 rayDirection = curr_pixel;
            
            // Perform raytracing
            glm::vec3 colour(0);
            Interval<float> tRange(0.f, std::numeric_limits<float>::infinity(), true, false);
            if (m_jitter == 0) {
                const Ray ray(
                    vinv * glm::vec4(0, 0, 0, 1),
                    glm::normalize(vinv * glm::vec4(rayDirection, 0))
                );
                colour = raycast(ray, std::move(tRange), scene, *opt_max_depth);
            } else {
                for (uint32_t i = 0; i < m_jitter; ++i) {
                    const float randomX = randomFloat() - 0.5f;
                    const float randomY = randomFloat() - 0.5f;
                    rayDirection = curr_pixel + randomX * pixel_dx + randomY * pixel_dy;
                    const Ray ray(
                        vinv * glm::vec4(0, 0, 0, 1),
                        glm::normalize(vinv * glm::vec4(rayDirection, 0))
                    );
                    colour = colour + raycast(ray, std::move(tRange), scene, *opt_max_depth);
                }

                colour = colour / static_cast<float>(m_jitter);
            }

            image[y][x] = vec2Pixel(colour);
        }
    }
    std::clog << "\rRendering completed: " << height << " / " << height << std::flush << std::endl;

    return image;
}