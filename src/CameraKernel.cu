#include <curand_kernel.h>
#include "Camera.cuh"
#include "CudaDeviceVec.cuh"
#include "CudaSerializable.cuh"
#include "Image.h"
#include "Random.cuh"
#include "World.cuh"

using namespace PPCast;

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
    VectorRef<Material>* __restrict__ materials,
    VectorRef<GeometryNode>* __restrict__ geometry,
    const Camera* __restrict__ camera, curandState* __restrict__ randomState
) {
    // Compute pixel index
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    const int pixelIndex = y * width + x;

    // Compute VectorRef data pointers
    materials->data = reinterpret_cast<Material*>    (reinterpret_cast<VectorRef<Material>*>(materials) + 1);
    geometry ->data = reinterpret_cast<GeometryNode*>(reinterpret_cast<VectorRef<GeometryNode>*>(geometry) + 1);
    World world(*materials, *geometry);

    // Generate ray
    RandomState rs(&randomState[pixelIndex]);
    glm::vec3 colour = camera->renderPixel(x, y, world, rs);

    // Write colour to framebuffer
    frameBuffer[pixelIndex] = {colour.x, colour.y, colour.z};
}

bool Camera::renderImageGPU(Image& image, const World& world) const {
    // Allocate output data buffers
    const uint32_t numPixels = width * height;
    CudaDeviceVec<float3>      d_frameBuffer(numPixels);
    CudaDeviceVec<curandState> d_randState  (numPixels);

    // Allocate input data buffers
    CudaDeviceBox<VectorRef<Material>>     d_materials  (world.materials);
    CudaDeviceBox<VectorRef<GeometryNode>> d_geometry   (world.geometry);
    CudaDeviceVec<Camera>                  d_camera     (*this);

    // Upload data to device
    d_materials.copyToDevice();
    d_geometry .copyToDevice();
    d_camera   .copyToDevice();

    // Compute thread block dimensions
    constexpr uint32_t blockWidth  = 8;
    constexpr uint32_t blockHeight = 8;
    const dim3 blocks((width / blockWidth) + 1, (height / blockHeight) + 1);
    const dim3 threads(blockWidth, blockHeight);

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