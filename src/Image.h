#ifndef PPCAST_IMAGE_H
#define PPCAST_IMAGE_H

#include "Common.h"

namespace PPCast {
    /**
     * @brief A colour, represented by its red, green, and blue components
     * 
     */
    struct Colour {
        /// @brief The red, green, and blue components of the colour, represented as values between 0 and 1
        float r, g, b;

        /**
         * @brief Construct a new Colour object -- default to black
         * 
         */
        Colour() : r(0), g(0), b(0) {}

        /**
         * @brief Construct a new Colour object from red, green, and blue components
         * 
         * @param r_ The red component of the colour, represented as a value between 0 and 1
         * @param g_ The green component of the colour, represented as a value between 0 and 1
         * @param b_ The blue component of the colour, represented as a value between 0 and 1
         */
        Colour(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

        /**
         * @brief Construct a new Colour object from a vector of colour components
         * 
         * @param v The vector of red, green, and blue colour components, represented as values between 0 and 1
         */
        Colour(const glm::vec3& v) : r(v.r), g(v.g), b(v.b) {}
    };

    /**
     * @brief An image composed of a rectangular grid of pixels
     * 
     */
    class Image {
    private:
        /// @brief The colour data for each pixel in the image
        std::unique_ptr<Colour[]> m_data;

    public:
        /// @brief The width of the image in pixels
        const uint32_t width;

        /// @brief The height of the image in pixels
        const uint32_t height;

        /**
         * @brief Construct a new Image object
         * 
         * @param w The width of the image in pixels
         * @param h The width of the image in pixels
         */
        Image(uint32_t w, uint32_t h)
            : m_data(new Colour[w * h])
            , width(w)
            , height(h)
        {}

        /**
         * @brief Get a pointer to the image data
         * 
         * @return the image data
         */
        inline Colour* data() const { return m_data.get(); }

        /**
         * @brief Get the colour corresponding to a pixel
         * 
         * @param x The x index of the pixel
         * @param y The y index of the pixel
         * @return The colour corresponding to the pixel
         */
        inline Colour& get(uint32_t x, uint32_t y) const { return m_data[y * width + x]; }

        /**
         * @brief Write the image data to file
         * 
         * @param filename the name of the output file
         */
        void write(const std::string& filename);
    };
}

#endif