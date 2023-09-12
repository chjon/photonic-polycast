#ifndef PPCAST_CUDASERIALIZABLE_H
#define PPCAST_CUDASERIALIZABLE_H

#include "CUDACommon.cuh"

namespace PPCast {
    class CudaSerializable {
        /**
         * @brief Compute the total amount of memory required to serialize the object
         * 
         * @return the total size in bytes
         */
        virtual size_t numBytes() const = 0;

        /**
         * @brief Serialize the object and write it to the device
         * 
         * @param devicePtr a pointer to device memory
         */
        virtual void copyToDevice(void* devicePtr) const {}
    };

    template <typename T>
    class VectorRef
        // : public CudaSerializable // We don't want a vtable!
    {
    public:
        const T* data;
        uint32_t size;

        VectorRef(const std::vector<T>& vec)
            : data(vec.data())
            , size(vec.size())
        {}

        __host__ __device__ VectorRef(const VectorRef& vec)
            : data(vec.data)
            , size(vec.size)
        {}

        __host__ __device__ inline const T& operator[] (uint32_t i) const { assert(i < size); return data[i]; }

        //////////////////////////////
        // CudaSerializable methods //
        //////////////////////////////

        __host__ __device__ inline size_t numBytes() const { return sizeof(VectorRef<T>) + sizeof(T) * size; }

        void copyToDevice(void* devicePtr) const {
            checkCudaErrors(cudaMemcpy(devicePtr, this, sizeof(VectorRef<T>), cudaMemcpyHostToDevice));
            devicePtr = reinterpret_cast<void*>(reinterpret_cast<VectorRef<T>*>(devicePtr) + 1);
            checkCudaErrors(cudaMemcpy(devicePtr, data, sizeof(T) * size, cudaMemcpyHostToDevice));
        }
    };

    template <typename T>
    class CudaDeviceBox {
    private:
        const T& m_hostData;
        T* m_deviceData;

    public:
        CudaDeviceBox(const T& data)
            : m_hostData(data)
            , m_deviceData(nullptr)
        {
            checkCudaErrors(cudaMalloc((void**)&m_deviceData, m_hostData.numBytes()));
        }

        inline void copyToDevice() { m_hostData.copyToDevice(reinterpret_cast<void*>(m_deviceData)); }

        inline T* get() const { return m_deviceData; }
    };
}

#endif