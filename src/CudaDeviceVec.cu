#include <cstdio>
#include "CudaDeviceVec.cuh"

using namespace PPCast;

template <typename T>
CudaDeviceVec<T>::CudaDeviceVec(unsigned int size)
    : m_data(nullptr)
    , m_size(size)
{
    if (cudaMalloc((void**)&m_data, sizeof(T) * m_size) != cudaSuccess) m_size = 0;
}

template <typename T>
CudaDeviceVec<T>::CudaDeviceVec(const std::vector<T>& data)
    : m_data(nullptr)
    , m_size(0)
{
    if (cudaMalloc((void**)&m_data, sizeof(T) * m_size) != cudaSuccess) return;

    if (cudaMemcpy(m_data, data.data(), sizeof(T) * m_size, cudaMemcpyHostToDevice) == cudaSuccess) {
        m_size = data.size();
    } else {
        cudaFree(m_data);
        m_data = nullptr;
    }
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
template class CudaDeviceVec<float2>;
template class CudaDeviceVec<float3>;
template class CudaDeviceVec<float4>;