#include <curand_kernel.h>
#include "Camera.cuh"
#include "CudaDeviceVec.cuh"
#include "CudaSerializable.cuh"
#include "Image.h"
#include "Random.cuh"
#include "World.cuh"

__global__ void renderInit(
    curandState *randomState, int width, int height,
    uint32_t seed
) {
    // Compute pixel index
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;

    // Each thread gets the same seed with a different sequence number and no offset
    const int pixelIndex = y * width + x;
    curand_init(seed, pixelIndex, 0, &randomState[pixelIndex]);
}

__global__ void renderImageGPUKernel(
    float3 *frameBuffer, int width, int height,
    PPCast::VectorRef<PPCast::Material>* materials,
    PPCast::VectorRef<PPCast::GeometryNode>* geometry,
    PPCast::Camera* camera, curandState *randomState
) {
    // Compute pixel index
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    const int pixelIndex = y * width + x;

    // Compute VectorRef data pointers
    materials->data = reinterpret_cast<PPCast::Material*>    (reinterpret_cast<PPCast::VectorRef<PPCast::Material>*>(materials) + 1);
    geometry ->data = reinterpret_cast<PPCast::GeometryNode*>(reinterpret_cast<PPCast::VectorRef<PPCast::GeometryNode>*>(geometry) + 1);
    PPCast::World world(*materials, *geometry);

    // Generate ray
    PPCast::RandomState rs(&randomState[pixelIndex]);
    glm::vec3 colour = camera->renderPixel(x, y, world, rs);

    // Write colour to framebuffer
    frameBuffer[pixelIndex] = {colour.x, colour.y, colour.z};
}

__host__ __device__ glm::vec3 PPCast::Camera::renderPixel(
    uint32_t x, uint32_t y,
    const PPCast::World& world,
    PPCast::RandomState& randomState
) const {
    // Perform raytracing
    glm::vec3 total(0);
    PPCast::Interval<float> tRange(0.f, INFINITY, true, false);
    for (uint32_t i = 0; i < raysPerPx; ++i) {
        const Ray ray = generateRay(x, y, randomState);
        total += raycast(ray, std::move(tRange), world, randomState);
    }
    return total / static_cast<float>(raysPerPx);
}

__host__ __device__ PPCast::Ray PPCast::Camera::generateRay(uint32_t x, uint32_t y, PPCast::RandomState& randomState) const {
    // Compute origin point in viewspace
    glm::vec3 rayOrigin = {0, 0, 0}; 

    // Perturb the origin point
    rayOrigin += m_defocusRadius * glm::vec3(randomInUnitSphere<2>(randomState), 0);

    // Compute the pixel sample position in the viewspace focal plane
    glm::vec3 pixelSample =
        m_pixel_topLeft +
        (static_cast<float>(x) * m_pixel_dx) +
        (static_cast<float>(y) * m_pixel_dy);
    
    // Perturb the pixel sample position
    pixelSample += (m_pixel_dx + m_pixel_dy) * glm::vec3(randomFloatVector<2>(randomState) - 0.5f * glm::vec2(1), 0);

    // Compute the pixel sample position in the worldspace focal plane
    glm::vec4 worldspacePixelSample = m_v2w * glm::vec4(pixelSample, 1);

    // Return ray in worldspace
    const glm::vec4 worldspaceRayOrigin = m_v2w * glm::vec4(rayOrigin, 1);
    const glm::vec4 worldspaceRayDir    = glm::normalize(worldspacePixelSample - worldspaceRayOrigin);
    return PPCast::Ray(worldspaceRayOrigin, worldspaceRayDir);
}

__host__ __device__ static glm::vec3 getSkyboxColour(const glm::vec4& direction) {
    constexpr glm::vec3 skyTopColour    = glm::vec3(0.5, 0.7, 1.0);
    constexpr glm::vec3 skyBottomColour = glm::vec3(1.0, 1.0, 1.0);
    const float a = 0.5f * (glm::normalize(direction).y + 1.f);
    return (1.f - a) * skyBottomColour + a * skyTopColour;
}

__host__ __device__ glm::vec3 PPCast::Camera::raycast(
    const Ray& ray, Interval<float>&& tRange,
    const PPCast::World& world, PPCast::RandomState& randomState
) const {
    PPCast::Ray currentRay = ray;
    glm::vec3 totalAttenuation(1);

    uint32_t maxDepth = maxBounces;
    while (maxDepth--) {
        // Check if ray intersects with geometry
        PPCast::HitInfo hitInfo;
        const bool hit = world.getIntersection(hitInfo, currentRay, tRange);

        // Return the sky colour if there is no intersection
        if (!hit) return totalAttenuation * getSkyboxColour(currentRay.direction());
        
        // Generate a reflected or transmitted ray
        glm::vec4 scatterDirection;
        glm::vec3 attenuation;
        assert(static_cast<uint32_t>(hitInfo.materialID) != UINT32_MAX);
        const PPCast::Material& material = world.materials[static_cast<uint32_t>(hitInfo.materialID)];
        const bool generatedRay = material.scatter(
            scatterDirection,
            attenuation,
            currentRay.direction(),
            hitInfo,
            randomState
        );
        
        // Return the material colour if there is no generated ray
        if (!generatedRay) return totalAttenuation * attenuation;

        // Cast the generated ray
        tRange.lower = 1e-3f;
        currentRay = PPCast::Ray(hitInfo.hitPoint, glm::normalize(scatterDirection));
        totalAttenuation = totalAttenuation * attenuation;
    }

    return glm::vec3(0);
}

bool PPCast::Camera::renderImageGPU(Image& image, const PPCast::World& world) const {
    // Allocate device buffers
    unsigned int numPixels = width * height;
    PPCast::CudaDeviceVec<float3>               d_frameBuffer(numPixels);
    PPCast::CudaDeviceVec<curandState>          d_randState  (numPixels);
    PPCast::CudaDeviceBox<PPCast::VectorRef<PPCast::Material>>     d_materials(world.materials);
    PPCast::CudaDeviceBox<PPCast::VectorRef<PPCast::GeometryNode>> d_geometry (world.geometry);
    PPCast::CudaDeviceVec<PPCast::Camera>       d_camera     (*this);

    // Upload data to device
    d_materials.copyToDevice();
    d_geometry .copyToDevice();
    d_camera   .copyToDevice();

    // Compute thread block dimensions
    int tx = 8;
    int ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx,ty);

    // Initialize
    renderInit<<<blocks, threads>>>(
        d_randState.get(), width, height, seed
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render image
    renderImageGPUKernel<<<blocks, threads>>>(
        d_frameBuffer.get(), width, height,
        d_materials.get(), d_geometry.get(),
        d_camera.get(), d_randState.get()
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy image back to host
    d_frameBuffer.copyToHost(reinterpret_cast<float3*>(image.data()));

    return true;
}