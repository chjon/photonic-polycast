#ifndef PPCAST_CUDADEVICEVEC_H
#define PPCAST_CUDADEVICEVEC_H

#include "CUDACommon.cuh"

namespace PPCast {
    template <typename T>
    class CudaDeviceVec {
    private:
        T* m_data;
        unsigned int m_size;
    public:
        CudaDeviceVec(unsigned int size);
        CudaDeviceVec(const CudaDeviceVec&) = delete;
        CudaDeviceVec(CudaDeviceVec&&) = default;
        ~CudaDeviceVec();

        CudaDeviceVec& operator=(const CudaDeviceVec&) = delete;
        CudaDeviceVec& operator=(CudaDeviceVec&&) = default;

        bool copyToDevice(const T* hostData, unsigned int size);
        bool copyToDevice(const T* hostData);
        bool copyToHost(T* hostData, unsigned int size) const;
        bool copyToHost(T* hostData) const;

        T* get() const;
        unsigned int size() const;
    };

    template <typename T>
    CudaDeviceVec<T>::CudaDeviceVec(unsigned int size)
        : m_data(nullptr)
        , m_size(0)
    {
        cudaError_t ret;
        checkCudaErrors(ret = cudaMalloc((void**)&m_data, sizeof(T) * size));
        if (ret == cudaSuccess) m_size = size;
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

    template <typename T>
    inline bool CudaDeviceVec<T>::copyToDevice(const T* hostData) { return copyToDevice(hostData, m_size); }

    template <typename T>
    inline bool CudaDeviceVec<T>::copyToHost(T* hostData) const { return copyToHost(hostData, m_size); }

    template <typename T>
    inline T* CudaDeviceVec<T>::get() const { return (m_size > 0) ? (m_data) : (nullptr); }

    template <typename T>
    inline unsigned int CudaDeviceVec<T>::size() const { return m_size; }
}

#endif