#include "Camera.h"
#include "Options.h"
#include "Ray.h"

using namespace PPCast;

static FloatOption opt_cam_fovy   ("cam-fovy"  , "the camera's y-axis FOV in degrees"          ,  90.0f);
static FloatOption opt_cam_aspect ("cam-aspect", "the camera's aspect ratio (x / y)"           ,   1.0f);
static FloatOption opt_cam_near   ("cam-near"  , "distance to the camera's near clipping plane",   0.1f);
static FloatOption opt_cam_far    ("cam-far"   , "distance to the camera's far clipping plane" , 100.0f);

Camera::Camera(const glm::vec3& pos, const glm::vec3& centre, const glm::vec3& up)
    : m_pos      (pos)
    , m_centre   (centre)
    , m_up       (up)
    , m_fov_y    (*opt_cam_fovy)
    , m_aspect   (*opt_cam_aspect)
    , m_nearPlane(*opt_cam_near)
    , m_farPlane (*opt_cam_far)
{}

static png::rgb_pixel raycast(const Ray& ray) {
    const glm::vec3 colour = 255.99f * 0.5f * (ray.direction() + glm::vec3(1, 1, 1));
    return png::rgb_pixel(colour.r, colour.g, colour.b);
}

png::image<png::rgb_pixel> Camera::render(uint32_t width, uint32_t height) const {
    // Compute viewport params
    const glm::vec3 viewport_x       = glm::vec3(1 * m_aspect,  0, 0);
    const glm::vec3 viewport_y       = glm::vec3(0, -1, 0);
    const glm::vec3 pixel_dx         = viewport_x / static_cast<float>(width);
    const glm::vec3 pixel_dy         = viewport_y / static_cast<float>(height);
    const glm::vec3 viewport_topLeft = glm::vec3(0, 0, m_nearPlane) - 0.5f * (viewport_x + viewport_y);
    const glm::vec3 pixel_topLeft    = viewport_topLeft + 0.5f * (pixel_dx + pixel_dy);

    // Actually render the image
    png::image<png::rgb_pixel> image(width, height);
    for (png::uint_32 y = 0; y < height; ++y) {
        std::clog << "Rendering scanlines: " << (y + 1) << " / " << height << "\r" << std::flush;
        for (png::uint_32 x = 0; x < width; ++x) {
            const glm::vec3 curr_pixel = pixel_topLeft + (static_cast<float>(x) * pixel_dx) + (static_cast<float>(y) * pixel_dy);
            const Ray ray(glm::vec3(0, 0, 0), curr_pixel);

            // TODO: pass in the world and perform raytracing
            image[y][x] = raycast(ray);
        }
    }
    std::clog << "Rendering completed: " << height << " / " << height << std::flush << std::endl;

    return image;
}