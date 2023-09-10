#ifndef PPCAST_IMAGE_H
#define PPCAST_IMAGE_H

#include "Common.h"

namespace PPCast {
    struct Colour {
        float r, g, b;

        Colour() : r(0), g(0), b(0) {}
        Colour(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
        Colour(const glm::vec3& v) : r(v.r), g(v.g), b(v.b) {}
    };

    class Image {
    private:
        std::unique_ptr<Colour[]> m_data;

    public:
        const uint32_t width, height;

        Image(uint32_t w, uint32_t h)
            : m_data(new Colour[w * h])
            , width(w)
            , height(h)
        {}

        inline Colour* data() const { return m_data.get(); }
        inline Colour& get(uint32_t x, uint32_t y) const { return m_data[y * width + x]; }

        void write(const std::string& filename);
    };
}

#endif