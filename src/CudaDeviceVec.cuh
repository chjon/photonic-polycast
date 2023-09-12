#ifndef PPCAST_CUDADEVICEVEC_H
#define PPCAST_CUDADEVICEVEC_H

#include "CUDACommon.cuh"

namespace PPCast {
    template <typename T>
    class CudaDeviceVec {
    private:
        T* m_deviceData;
        const T* m_hostSrc;
        T* m_hostDest;
        size_t m_size;
    public:
        CudaDeviceVec(unsigned int size);
        CudaDeviceVec(T& val);
        CudaDeviceVec(const T& val);
        CudaDeviceVec(const CudaDeviceVec&) = delete;
        CudaDeviceVec(CudaDeviceVec&&) = default;
        ~CudaDeviceVec();

        CudaDeviceVec& operator=(const CudaDeviceVec&) = delete;
        CudaDeviceVec& operator=(CudaDeviceVec&&) = default;

        bool copyToDevice(const T* hostData, unsigned int size);
        bool copyToDevice(const T* hostData);
        bool copyToDevice();
        bool copyToHost(T* hostData, unsigned int size) const;
        bool copyToHost(T* hostData) const;
        bool copyToHost() const;

        T* get() const;
        unsigned int size() const;
    };

    template <typename T>
    CudaDeviceVec<T>::CudaDeviceVec(unsigned int size)
        : m_deviceData(nullptr)
        , m_size(0)
    {
        cudaError_t ret;
        checkCudaErrors(ret = cudaMalloc((void**)&m_deviceData, sizeof(T) * size));
        if (ret == cudaSuccess) m_size = sizeof(T) * size;
    }

    template <typename T>
    CudaDeviceVec<T>::CudaDeviceVec(T& val)
        : m_deviceData(nullptr)
        , m_hostSrc(&val)
        , m_hostDest(&val)
        , m_size(0)
    {
        cudaError_t ret;
        checkCudaErrors(ret = cudaMalloc((void**)&m_deviceData, sizeof(T)));
        if (ret == cudaSuccess) m_size = sizeof(T);
    }

    template <typename T>
    CudaDeviceVec<T>::CudaDeviceVec(const T& val)
        : m_deviceData(nullptr)
        , m_hostSrc(&val)
        , m_hostDest(nullptr)
        , m_size(0)
    {
        cudaError_t ret;
        checkCudaErrors(ret = cudaMalloc((void**)&m_deviceData, sizeof(T)));
        if (ret == cudaSuccess) m_size = sizeof(T);
    }

    template <typename T>
    CudaDeviceVec<T>::~CudaDeviceVec() {
        if (m_size) cudaFree(m_deviceData);
    }

    template <typename T>
    bool CudaDeviceVec<T>::copyToDevice(const T* hostData, unsigned int size) {
        cudaError_t ret;
        checkCudaErrors(ret = cudaMemcpy(m_deviceData, hostData, sizeof(T) * size, cudaMemcpyHostToDevice));
        return ret == cudaSuccess;
    }

    template <typename T>
    bool CudaDeviceVec<T>::copyToHost(T* hostData, unsigned int size) const {
        cudaError_t ret;
        checkCudaErrors(ret = cudaMemcpy(hostData, m_deviceData, sizeof(T) * size, cudaMemcpyDeviceToHost));
        return ret == cudaSuccess;
    }

    template <typename T>
    inline bool CudaDeviceVec<T>::copyToDevice(const T* hostData) {
        cudaError_t ret;
        checkCudaErrors(ret = cudaMemcpy(m_deviceData, hostData, m_size, cudaMemcpyHostToDevice));
        return ret == cudaSuccess;
    }

    template <typename T>
    inline bool CudaDeviceVec<T>::copyToHost(T* hostData) const {
        cudaError_t ret;
        checkCudaErrors(ret = cudaMemcpy(hostData, m_deviceData, m_size, cudaMemcpyDeviceToHost));
        return ret == cudaSuccess;
    }

    template <typename T>
    inline bool CudaDeviceVec<T>::copyToDevice() {
        return copyToDevice(m_hostSrc);
    }

    template <typename T>
    inline bool CudaDeviceVec<T>::copyToHost() const {
        return copyToHost(m_hostDest);
    }

    template <typename T>
    inline T* CudaDeviceVec<T>::get() const { return (m_size > 0) ? (m_deviceData) : (nullptr); }

    template <typename T>
    inline unsigned int CudaDeviceVec<T>::size() const { return m_size; }
}

#endif