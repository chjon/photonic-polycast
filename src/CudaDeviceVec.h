#ifndef PPCAST_CUDADEVICEVEC_H
#define PPCAST_CUDADEVICEVEC_H

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
    };

    template <typename T>
    inline bool CudaDeviceVec<T>::copyToDevice(const T* hostData) { return copyToDevice(hostData, m_size); }

    template <typename T>
    inline bool CudaDeviceVec<T>::copyToHost(T* hostData) const { return copyToHost(hostData, m_size); }

    template <typename T>
    inline T* CudaDeviceVec<T>::get() const { return m_data; }
}

#endif