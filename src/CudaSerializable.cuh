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
        virtual void copyToDevice(void* devicePtr) const = 0;
    };

    /**
     * @brief A class which references vector data
     * NB: this class does not have ownership of the vector data
     * 
     * @tparam T the type of data stored in the vector
     */
    template <typename T>
    class VectorRef
        // : public CudaSerializable // We don't want a vtable!
    {
    public:
        /// @brief The vector data
        const T* data;

        /// @brief The number of elements in the vector
        size_t size;

        /**
         * @brief Construct a new VectorRef referencing the data in a std::vector
         * 
         * @param vec the std::vector to reference
         */
        VectorRef(const std::vector<T>& vec)
            : data(vec.data())
            , size(vec.size())
        {}

        /**
         * @brief Copy a VectorRef
         * 
         * @param vec the VectorRef to copy
         */
        __host__ __device__ VectorRef(const VectorRef& vec)
            : data(vec.data)
            , size(vec.size)
        {}

        /**
         * @brief A convenience operator for accessing data in the vector
         * 
         * @param i The index to access
         * @return The vector data at index i
         */
        __host__ __device__ inline const T& operator[] (uint32_t i) const {
            assert(i < size);
            return data[i];
        }

        //////////////////////////////
        // CudaSerializable methods //
        //////////////////////////////

        /**
         * @brief Get the number of bytes required to store the object and its data
         * 
         * @return The number of bytes required to store the object and its data
         */
        __host__ __device__ inline size_t numBytes() const { return sizeof(VectorRef<T>) + sizeof(T) * size; }

        /**
         * @brief Copy the object and its data to the device
         * 
         * @param devicePtr A pointer to at least @code{numBytes()} memory on the device
         */
        void copyToDevice(void* devicePtr) const {
            checkCudaErrors(cudaMemcpy(devicePtr, this, sizeof(VectorRef<T>), cudaMemcpyHostToDevice));
            devicePtr = reinterpret_cast<void*>(reinterpret_cast<VectorRef<T>*>(devicePtr) + 1);
            checkCudaErrors(cudaMemcpy(devicePtr, data, sizeof(T) * size, cudaMemcpyHostToDevice));
        }
    };

    /**
     * @brief A class for managing device memory to store an object
     * 
     * @tparam The type of object to store
     */
    template <typename T>
    class CudaDeviceBox {
    private:
        /// @brief The host data
        const T& m_hostData;

        /// @brief A pointer to device memory allocated for the data
        T* m_deviceData;

    public:
        /**
         * @brief Construct a new CudaDeviceBox, allocating memory on the device
         * 
         * @param data 
         */
        CudaDeviceBox(const T& data)
            : m_hostData(data)
            , m_deviceData(nullptr)
        {
            checkCudaErrors(cudaMalloc((void**)&m_deviceData, m_hostData.numBytes()));
        }

        /**
         * @brief Destroy the CudaDeviceBox, freeing the memory on the device
         * 
         */
        ~CudaDeviceBox() {
            if (m_deviceData != nullptr) {
                checkCudaErrors(cudaFree(m_deviceData));
            }
        }

        // Delete copy assignment operator
        // Only one CudaDeviceBox can have ownership of the device pointer at a time
        CudaDeviceBox<T>& operator= (const CudaDeviceBox<T>& other) = delete;

        /**
         * @brief Copy the host data to the device
         * 
         */
        inline void copyToDevice() { m_hostData.copyToDevice(reinterpret_cast<void*>(m_deviceData)); }

        /**
         * @brief Get a pointer to the memory allocated on the device
         * 
         * @return a pointer to the memory allocated on the device
         */
        inline T* get() const { return m_deviceData; }
    };
}

#endif