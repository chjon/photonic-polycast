#ifndef PPCAST_TYPES_H
#define PPCAST_TYPES_H

#include "Common.h"

namespace PPCast {
    struct MaterialID {
    private:
        uint32_t m_value;
    
        MaterialID(uint32_t val) : m_value(val) {}

    public:
        static uint32_t numMaterials;
        static const MaterialID INVALID;

        MaterialID() = delete;
        MaterialID(const MaterialID& material) : m_value(material.m_value) {}
        MaterialID(MaterialID&& material)      : m_value(material.m_value) {}

        inline MaterialID& operator=(const MaterialID& material) { m_value = material.m_value; return *this; }
        inline MaterialID& operator=(MaterialID&& material)      { m_value = material.m_value; return *this; }

        inline bool operator==(const MaterialID& material) { return m_value == material.m_value; }
        inline bool operator!=(const MaterialID& material) { return !operator==(material); }

        inline explicit operator uint32_t() const { return m_value; }

        static inline MaterialID next() { return MaterialID(numMaterials++); }
    };
}

#endif