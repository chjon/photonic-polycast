#include "Camera.h"
#include "Options.h"
#include "Ray.h"

using namespace PPCast;

static FloatOption opt_cam_fovy   ("cam-fovy"  , "the camera's y-axis FOV in degrees",  90.0f);
static FloatOption opt_cam_aspect ("cam-aspect", "the camera's aspect ratio (x / y)" ,   1.0f);

Camera::Camera(const glm::vec3& pos, const glm::vec3& centre, const glm::vec3& up)
    : m_pos   (pos)
    , m_centre(centre)
    , m_up    (up)
    , m_fovy  (*opt_cam_fovy)
    , m_aspect(*opt_cam_aspect)
{}

static png::rgb_pixel vec2colour(const glm::vec3& v) {
    const glm::vec3 colour = 255.99f * 0.5f * (glm::normalize(v) + glm::vec3(1));
    return png::rgb_pixel(
        static_cast<unsigned char>(colour.r),
        static_cast<unsigned char>(colour.g),
        static_cast<unsigned char>(colour.b)
    );
}

static png::rgb_pixel raycast(const std::vector<GeometryNode>& scene, const Ray& ray) {
    HitInfo hitInfo;
    for (const GeometryNode& node : scene) {
        if (node.getIntersection(hitInfo, ray)) {
            return vec2colour(hitInfo.normal);
        }
    }

    const float a = 0.5f * (glm::normalize(ray.direction()).y + 1.f);
    return vec2colour((1.f - a) * glm::vec3(1.f) + a * glm::vec3(0.5, 0.7, 1.0));
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

    // Actually render the image
    png::image<png::rgb_pixel> image(width, height);
    for (png::uint_32 y = 0; y < height; ++y) {
        std::clog << "\rRendering scanlines: " << (y + 1) << " / " << height << " " << std::flush;
        for (png::uint_32 x = 0; x < width; ++x) {
            const glm::vec3 curr_pixel = pixel_topLeft + (static_cast<float>(x) * pixel_dx) + (static_cast<float>(y) * pixel_dy);
            const glm::mat4 vinv = glm::inverse(getView());
            const Ray ray(
                vinv * glm::vec4(0, 0, 0, 1),
                vinv * glm::vec4(curr_pixel, 0),
                Interval<float>(0.f, std::numeric_limits<float>::infinity(), true, false)
            );

            // Perform raytracing
            image[y][x] = raycast(scene, ray);
        }
    }
    std::clog << "\rRendering completed: " << height << " / " << height << std::flush << std::endl;

    return image;
}