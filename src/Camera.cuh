#ifndef PPCAST_CAMERA_H
#define PPCAST_CAMERA_H

#include "Common.h"
#include "Ray.cuh"
#include "SceneNode.cuh"
#include "World.cuh"

namespace PPCast {
    class Image;
    union RandomState;
    class World;

    class Camera {
    private:
        ////////////////////
        // Derived values //
        ////////////////////
        // These values are computed once and cached for generating rays
        
        /// @brief The horizontal size of a pixel in viewspace
        glm::vec3 m_pixel_dx;

        /// @brief The vertical size of a pixel in viewspace
        glm::vec3 m_pixel_dy;

        /// @brief The top-left corner of the focal plane in viewspace
        glm::vec3 m_pixel_topLeft;

        /// @brief The radius of the defocus disk
        float m_defocusRadius;
        
        /// @brief The viewspace-to-worldspace transformation matrix
        glm::mat4 m_v2w;

        /////////////////
        // Private API //
        /////////////////

        /**
         * @brief Generate an image from a given scene via raytracing on the CPU
         * 
         * @param image The image to output
         * @param scene The scene to render
         * @return true if the image was rendered successfully
         */
        bool renderImageCPU(Image& image, const World& scene) const;

        /**
         * @brief Generate an image from a given scene via raytracing on the GPU
         * 
         * @param image The image to output
         * @param scene The scene to render
         * @return true if the image was rendered successfully
         */
        bool renderImageGPU(Image& image, const World& scene) const;

        /**
         * @brief Sample a ray to cast for a pixel
         * 
         * @param x The x position of the pixel
         * @param y The y position of the pixel
         * @param randomState The state for random number generation
         * @return A randomly sampled ray for the pixel
         */
        __host__ __device__ Ray generateRay(uint32_t x, uint32_t y, RandomState& randomState) const;

        /**
         * @brief Perform raytracing for a single ray
         * 
         * @param ray The ray to trace
         * @param tRange The permissible scale values of the ray's direction vector
         * @param world The scene to render
         * @param randomState The state for random number generation
         * @return the colour contribution of the ray 
         */
        __host__ __device__ glm::vec3 raycast(
            const Ray& ray, Interval<float>&& tRange,
            const World& world, RandomState& randomState
        ) const;

    public:
        /////////////////////////////////////
        // Camera position and orientation //
        /////////////////////////////////////

        /// @brief Camera centre ("eye" position)
        glm::vec3 lookfrom;

        /// @brief Camera direction
        glm::vec3 lookat;

        /// @brief Camera orientation
        glm::vec3 up;

        //////////////////////////
        // Camera lens settings //
        //////////////////////////
        
        /// @brief Vertical field-of-view, measured in degrees
        float vfov;

        /// @brief Depth of field, measured in degrees
        float dofAngle;

        /// @brief Focal distance
        float focalDist;

        //////////////////////////////////
        // Image dimensions and quality //
        //////////////////////////////////

        /// @brief Image width in pixels
        uint32_t width;

        /// @brief Image height in pixels
        uint32_t height;

        /// @brief Number of rays to cast per pixel
        uint32_t raysPerPx;

        /// @brief Maximum number of ray bounces
        uint32_t maxBounces;

        //////////////////////////
        // Computation settings //
        //////////////////////////

        /// @brief Seed for pseudorandom number generation
        uint32_t seed;

        /// @brief Whether to use the GPU for rendering
        bool useGPU;

        /**
         * @brief Default camera constructor -- constructs camera using commandline options
         * 
         */
        Camera();

        /**
         * @brief Compute derived values from public parameters in preparation for raytracing
         * 
         * @param width The width of the image
         * @param height The height of the image
         */
        void initialize(uint32_t width, uint32_t height);

        /**
         * @brief Generate an image from a given scene via raytracing
         * 
         * @param image The image to output
         * @param scene The scene to render
         * @return true if the image was rendered successfully
         */
        bool renderImage(Image& image, const World& scene) const;

        /**
         * @brief Render a pixel of a given scene via raytracing
         * 
         * @param x The pixel's x coordinate
         * @param y The pixel's y coordinate
         * @param world The scene to render
         * @param randomState The state data to use for random number generation
         * @return the colour of the pixel
         * 
         * @note This is public because it needs to be accessible to the GPU kernel
         */
        __host__ __device__ glm::vec3 renderPixel(
            uint32_t x, uint32_t y,
            const World& world,
            RandomState& randomState
        ) const;
    };

    ////////////////////////////////////
    // Inline function implementation //
    ////////////////////////////////////

    inline bool Camera::renderImage(Image& image, const World& scene) const {
        if (useGPU) return renderImageGPU(image, scene);
        else        return renderImageCPU(image, scene);
    }
}

#endif