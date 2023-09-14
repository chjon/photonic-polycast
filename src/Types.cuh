#ifndef PPCAST_TYPES_H
#define PPCAST_TYPES_H

#include "Common.h"

namespace PPCast {
    /**
     * @brief A unique identifier for a material
     * 
     * @note This is just a counter
     */
    struct MaterialID {
    private:
        /// @brief The integer value of the identifier
        /// @note if this value is @code{UINT32_MAX}, then the material ID is invalid
        uint32_t m_value;

        /**
         * @brief Create a new MaterialID
         * 
         * @param val The value of the material's ID
         */
        __host__ __device__ MaterialID(uint32_t val) : m_value(val) {}

    public:
        /// @brief The current total number of materials
        static uint32_t numMaterials;

        /**
         * @brief Create a new, invalid MaterialID object
         * 
         */
        __host__ __device__ MaterialID() : m_value(UINT32_MAX) {}

        /**
         * @brief Copy a MaterialID
         * 
         * @param material the MaterialID to copy
         */
        __host__ __device__ MaterialID(const MaterialID& material) : m_value(material.m_value) {}

        /**
         * @brief Move a MaterialID
         * 
         * @param material the MaterialID to take ownership of
         */
        __host__ __device__ MaterialID(MaterialID&& material) : m_value(material.m_value) {}

        /**
         * @brief Copy a MaterialID
         * 
         * @param material the MaterialID to copy
         * @return the MaterialID
         */
        __host__ __device__ inline MaterialID& operator=(const MaterialID& material) { m_value = material.m_value; return *this; }

        /**
         * @brief Move a MaterialID
         * 
         * @param material the MaterialID to take ownership of
         * @return the MaterialID
         */
        __host__ __device__ inline MaterialID& operator=(MaterialID&& material)      { m_value = material.m_value; return *this; }

        /**
         * @brief Check MaterialIDs for equality
         * 
         * @param material the MaterialID to check against
         * @return true if the MaterialIDs are the same
         */
        __host__ __device__ inline bool operator==(const MaterialID& material) { return m_value == material.m_value; }

        /**
         * @brief Check MaterialIDs for inequality
         * 
         * @param material the MaterialID to check against
         * @return true if the MaterialIDs are different
         */
        __host__ __device__ inline bool operator!=(const MaterialID& material) { return !operator==(material); }

        /**
         * @brief Get the MaterialID's internal value
         * 
         * @return uint32_t the MaterialID's internal value
         */
        __host__ __device__ inline explicit operator uint32_t() const { return m_value; }

        /**
         * @brief Get a new, unique MaterialID -- this is the canonical way to instantiate valid
         * MaterialIDs
         * 
         * @return A new, unique MaterialID
         */
        static inline MaterialID next() { return MaterialID(numMaterials++); }
    };
}

#endif