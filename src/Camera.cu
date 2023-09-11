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
{}

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
    m_v2w           = glm::inverse(glm::lookAt(pos, centre, up));
    m_defocusRadius = focalDist * glm::tan(glm::radians(0.5f * dofAngle));
}

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

bool Camera::renderImage(Image& image, const World& scene) const {
    if (*opt_usegpu) return renderImageGPU(image, scene);
    else             return renderImageCPU(image, scene);
}