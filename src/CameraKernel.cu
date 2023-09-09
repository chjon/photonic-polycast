#include "Camera.h"
#include "CudaDeviceVec.cuh"
#include "Image.h"
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

__global__ void render(float3 *frameBuffer, int width, int height) {
    // Compute pixel index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int pixel_index = y * width + x;

    frameBuffer[pixel_index] = {
        static_cast<float>(x) / width,
        static_cast<float>(y) / height,
        0.2f
    };
}

__host__ __device__ PPCast::Ray PPCast::Camera::generateRay(uint32_t x, uint32_t y) const {
    // Compute origin point in viewspace
    glm::vec3 rayOrigin = {0, 0, 0}; 

    // Perturb the origin point
    rayOrigin += m_defocusRadius * glm::vec3(randomInUnitSphere<2>(), 0);

    // Compute the pixel sample position in the viewspace focal plane
    glm::vec3 pixelSample =
        m_pixel_topLeft +
        (static_cast<float>(x) * m_pixel_dx) +
        (static_cast<float>(y) * m_pixel_dy);
    
    // Perturb the pixel sample position
    pixelSample += (m_pixel_dx + m_pixel_dy) * glm::vec3(randomFloatVector<2>() - 0.5f * glm::vec2(1), 0);

    // Compute the pixel sample position in the worldspace focal plane
    glm::vec4 worldspacePixelSample = m_v2w * glm::vec4(pixelSample, 1);

    // Return ray in worldspace
    const glm::vec4 worldspaceRayOrigin = m_v2w * glm::vec4(rayOrigin, 1);
    const glm::vec4 worldspaceRayDir    = glm::normalize(worldspacePixelSample - worldspaceRayOrigin);
    return PPCast::Ray(worldspaceRayOrigin, worldspaceRayDir);
}

bool PPCast::Camera::renderImageGPU(Image& image, const PPCast::World& scene) const {
    unsigned int numPixels = width * height;
    PPCast::CudaDeviceVec<float3> d_frameBuffer(numPixels);
    if (d_frameBuffer.size() == 0) {
        std::cerr << "failed to allocate GPU frameBuffer" << std::endl;
        return false;
    } 
    
    // Upload data to device
    // PPCast::CudaDeviceVec<GeometryNode> d_geometry(scene.getGeometry());

    // Compute thread block dimensions
    int tx = 8;
    int ty = 8;

    // Render image
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx,ty);
    render<<<blocks,threads>>>(d_frameBuffer.get(), width, height);

    // Copy image back to host
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    d_frameBuffer.copyToHost(reinterpret_cast<float3*>(image.data()));

    return true;
}