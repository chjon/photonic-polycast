#include "Common.h"
#include "CudaDeviceVec.h"

using namespace PPCast;

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int mainCUDA() {
    unsigned int n = 10;
    std::vector<float> a(n, 1.f);
    std::vector<float> b(n, 2.f);
    std::vector<float> o(n, 0.f);
    CudaDeviceVec<float> d_a(n);
    CudaDeviceVec<float> d_b(n);
    CudaDeviceVec<float> d_o(n);

    for (unsigned int i = 0; i < n; ++i)
        printf("%f,", a[i]);
    printf("\n");
    for (unsigned int i = 0; i < n; ++i)
        printf("%f,", b[i]);
    printf("\n");

    // Transfer data from host to device memory
    if (!d_a.copyToDevice(a.data())) return 1;
    if (!d_b.copyToDevice(b.data())) return 1;

    // Add vectors
    vector_add<<<1,1>>>(d_o.get(), d_a.get(), d_b.get(), n); 
    cudaDeviceSynchronize();

    // Transfer data from device to host memory
    d_o.copyToHost(o.data());

    // Output results
    for (unsigned int i = 0; i < n; ++i) printf("%f,", o[i]);
    printf("\n");

    return 0;
}