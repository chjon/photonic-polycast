#ifndef PPCAST_TYPES_H
#define PPCAST_TYPES_H

#include "Common.h"

namespace PPCast {
    struct MaterialID {
    private:
        static uint32_t numMaterials;
        uint32_t m_value;
    
    public:
        MaterialID() : m_value(numMaterials++) {}
        explicit operator uint32_t() const { return m_value; }
    };
}

#endif