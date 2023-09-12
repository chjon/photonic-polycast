#ifndef PPCAST_CAMERA_H
#define PPCAST_CAMERA_H

#include "Common.h"
#include "Ray.cuh"
#include "SceneNode.cuh"
#include "World.cuh"

namespace PPCast {
    class Image;
    class RandomState;
    class World;

    class Camera {
    private:
        // Derived values -- cached for generating rays
        glm::vec3 m_pixel_dx;
        glm::vec3 m_pixel_dy;
        glm::vec3 m_pixel_topLeft;
        glm::mat4 m_v2w;
        float     m_defocusRadius;

        bool renderImageCPU(Image& image, const World& scene) const;
        bool renderImageGPU(Image& image, const World& scene) const;

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

        /// @brief Seed for pseudorandom number generation
        uint32_t seed;

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
         * @brief Sample a ray to cast for a pixel
         * 
         * @param x The x position of the pixel
         * @param y The y position of the pixel
         * @param randomState The state for random number generation
         * @return A randomly sampled ray for the pixel
         */
        __host__ __device__ Ray generateRay(uint32_t x, uint32_t y, RandomState& randomState) const;

        __host__ __device__ static glm::vec3 raycast(
            const Ray& ray, Interval<float>&& tRange,
            const World& world,
            unsigned int maxDepth, RandomState& randomState
        );

        __host__ __device__ glm::vec3 renderPixel(
            uint32_t x, uint32_t y,
            const World& world,
            RandomState& randomState
        ) const;
    };
}

#endif