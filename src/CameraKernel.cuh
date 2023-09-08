#ifndef PPCAST_CAMERAKERNEL_CUH
#define PPCAST_CAMERAKERNEL_CUH

#include "Common.h"

#ifndef __NVCC__
    #define __device__
    #define __global__
#endif

void renderImageGPU(float *frameBuffer, unsigned int width, unsigned int height);

#endif