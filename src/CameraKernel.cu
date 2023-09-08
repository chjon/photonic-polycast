#include "CameraKernel.cuh"
#include "CudaDeviceVec.h"

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

void renderImageGPU(float *frameBuffer, unsigned int width, unsigned int height) {
    unsigned int numPixels = width * height;
    PPCast::CudaDeviceVec<float3> d_frameBuffer(numPixels);
    
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
    d_frameBuffer.copyToHost(reinterpret_cast<float3*>(frameBuffer));
}