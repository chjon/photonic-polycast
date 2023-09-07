#include <cstdio>
#include "CudaDeviceVec.h"

using namespace PPCast;

template <typename T>
CudaDeviceVec<T>::CudaDeviceVec(unsigned int size)
    : m_data(nullptr)
    , m_size(size)
{
    if (cudaMalloc((void**)&m_data, sizeof(T) * m_size) != cudaSuccess) m_size = 0;
}

template <typename T>
CudaDeviceVec<T>::~CudaDeviceVec() {
    if (m_size) cudaFree(m_data);
}

template <typename T>
bool CudaDeviceVec<T>::copyToDevice(const T* hostData, unsigned int size) {
    return cudaMemcpy(m_data, hostData, sizeof(T) * size, cudaMemcpyHostToDevice) == cudaSuccess;
}

template <typename T>
bool CudaDeviceVec<T>::copyToHost(T* hostData, unsigned int size) const {
    return cudaMemcpy(hostData, m_data, sizeof(T) * size, cudaMemcpyDeviceToHost) == cudaSuccess;
}

// Explicit class initialization
template class CudaDeviceVec<float>;