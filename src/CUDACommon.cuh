#ifndef PPCAST_CUDACOMMON_H
#define PPCAST_CUDACOMMON_H

#include "Common.h"

// CUDA utility macro
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

#endif