#include <png++/png.hpp>
#include "Image.h"

using namespace PPCast;

static png::rgb_pixel vec2Pixel(const Colour& c) {
    const glm::vec3 colour = 255.99f * glm::sqrt(glm::vec3(c.r, c.g, c.b));
    return png::rgb_pixel(
        static_cast<unsigned char>(colour.r),
        static_cast<unsigned char>(colour.g),
        static_cast<unsigned char>(colour.b)
    );
}

void Image::write(const std::string& filename) {
    png::image<png::rgb_pixel> pngImage(width, height);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            pngImage[y][x] = vec2Pixel(get(x, y));
        }
    }
    pngImage.write(filename);
}