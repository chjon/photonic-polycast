#include <curand_kernel.h>
#include "Camera.h"
#include "CudaDeviceVec.cuh"
#include "Image.h"
#include "Random.h"
#include "World.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

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
    PPCast::Material* materials, int numMaterials,
    PPCast::GeometryNode* geometry, int numGeometry,
    PPCast::Camera* camera, curandState *randomState
) {
    // Compute pixel index
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    const int pixelIndex = y * width + x;

    // Generate ray
    PPCast::RandomState rs(&randomState[pixelIndex]);
    glm::vec3 colour = camera->renderPixel(
        x, y,
        materials, numMaterials,
        geometry, numGeometry,
        rs
    );

    
    // Write colour to framebuffer
    frameBuffer[pixelIndex] = {
        static_cast<float>(x) / width,
        static_cast<float>(y) / height,
        0.2f
    };
}

__host__ __device__ glm::vec3 PPCast::Camera::renderPixel(
    uint32_t x, uint32_t y,
    const PPCast::Material* materials, size_t numMaterials,
    const PPCast::GeometryNode* geometry, size_t numGeometry,
    PPCast::RandomState& randomState
) const {
    // Perform raytracing
    glm::vec3 total(0);
    PPCast::Interval<float> tRange(0.f, INFINITY, true, false);
    for (uint32_t i = 0; i < raysPerPx; ++i) {
        const Ray ray = generateRay(x, y, randomState);
        total += Camera::raycast(
            ray, std::move(tRange),
            materials, numMaterials,
            geometry, numGeometry,
            maxBounces, randomState
        );
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
    const PPCast::Material* materials, size_t numMaterials,
    const PPCast::GeometryNode* geometry, size_t numGeometry,
    unsigned int maxDepth, PPCast::RandomState& randomState
) {
    PPCast::Ray currentRay = ray;
    glm::vec3 totalAttenuation(1);

    while (maxDepth--) {
        // Check if ray intersects with geometry
        PPCast::HitInfo hitInfo;
        const bool hit = World::getIntersection(
            hitInfo, currentRay, tRange,
            geometry, numGeometry
        );

        // Return the sky colour if there is no intersection
        if (!hit) return totalAttenuation * getSkyboxColour(currentRay.direction());
        
        // Generate a reflected or transmitted ray
        glm::vec4 scatterDirection;
        glm::vec3 attenuation;
        assert(static_cast<uint32_t>(hitInfo.materialID) != UINT32_MAX);
        const PPCast::Material& material = materials[static_cast<uint32_t>(hitInfo.materialID)];
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

bool PPCast::Camera::renderImageGPU(Image& image, const PPCast::World& scene) const {
    unsigned int numPixels = width * height;
    PPCast::CudaDeviceVec<float3> d_frameBuffer(numPixels);
    if (d_frameBuffer.size() == 0) {
        std::cerr << "failed to allocate d_frameBuffer" << std::endl;
        return false;
    }

    PPCast::CudaDeviceVec<curandState> d_randState(numPixels);
    if (d_randState.size() == 0) {
        std::cerr << "failed to allocate d_randState" << std::endl;
        return false;
    }
    
    // Upload data to device
    PPCast::CudaDeviceVec<PPCast::Material> d_materials(scene.getMaterials());
    if (d_materials.size() == 0) {
        std::cerr << "failed to allocate or upload d_materials" << std::endl;
        return false;
    }

    PPCast::CudaDeviceVec<PPCast::GeometryNode> d_geometry(scene.getGeometry());
    if (d_geometry.size() == 0) {
        std::cerr << "failed to allocate or upload d_geometry" << std::endl;
        return false;
    }

    PPCast::CudaDeviceVec<PPCast::Camera> d_camera(*this);
    if (d_geometry.size() == 0) {
        std::cerr << "failed to allocate or upload d_camera" << std::endl;
        return false;
    }

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
        d_materials.get(), d_materials.size(),
        d_geometry.get(), d_geometry.size(),
        d_camera.get(), d_randState.get()
    );

    // Copy image back to host
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    d_frameBuffer.copyToHost(reinterpret_cast<float3*>(image.data()));

    return true;
}