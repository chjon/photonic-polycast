#ifndef PPCAST_TYPES_H
#define PPCAST_TYPES_H

#include "Common.h"

namespace PPCast {
    struct MaterialID {
    private:
        uint32_t m_value;
    

    public:
        static uint32_t numMaterials;

        __host__ __device__ MaterialID(uint32_t val = UINT32_MAX)  : m_value(val) {}
        __host__ __device__ MaterialID(const MaterialID& material) : m_value(material.m_value) {}
        __host__ __device__ MaterialID(MaterialID&& material)      : m_value(material.m_value) {}

        __host__ __device__ inline MaterialID& operator=(const MaterialID& material) { m_value = material.m_value; return *this; }
        __host__ __device__ inline MaterialID& operator=(MaterialID&& material)      { m_value = material.m_value; return *this; }

        __host__ __device__ inline bool operator==(const MaterialID& material) { return m_value == material.m_value; }
        __host__ __device__ inline bool operator!=(const MaterialID& material) { return !operator==(material); }

        __host__ __device__ inline explicit operator uint32_t() const { return m_value; }

        static inline MaterialID next() { return MaterialID(numMaterials++); }
    };
}

#endif