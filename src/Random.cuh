#ifndef PPCAST_RANDOM_H
#define PPCAST_RANDOM_H

#include <curand_kernel.h>
#include <random>
#include "Common.h"

namespace PPCast {
    /**
     * @brief A wrapper for random state -- this is a union because the host and device use
     * different random number generators
     * 
     * @note The union stores pointers instead of the state itself since the object can be large,
     * and also because the device does not support the standard library's random number generators
     */
    union RandomState {
    private:
        /// @brief A pointer to the state used by CUDA's random number generator
        curandState* randomState;

        /// @brief A pointer to a random number generator
        std::mt19937* randomGenerator;

    public:
        /**
         * @brief Construct the RandomState on the device using CUDA random state
         * 
         * @param rs The CUDA random state
         */
        __device__ RandomState(curandState* rs) : randomState(rs) {}

        /**
         * @brief Construct the Random state on the host using standard library RNG
         * 
         * @param rg The standard library random number generator
         */
        __host__ RandomState(std::mt19937* rg) : randomGenerator(rg) {}

        /**
         * @brief Generate a random float in [0, 1)
         * 
         * @return a random float in [0, 1)
         */
        __host__ __device__ inline float random() {
            #ifdef __CUDA_ARCH__
            return 1.f - curand_uniform(randomState);
            #else
            std::mt19937& rg = *randomGenerator;
            return static_cast<float>(rg()) / (static_cast<float>(rg.max()) + 1.f);
            #endif
        }
    };

    ///////////////////////
    // Utility functions //
    ///////////////////////

    /**
     * @brief Generate a random float in [0, 1)
     * 
     * @param randomState the random state
     * 
     * @return a random float in [0, 1)
     */
    __host__ __device__ inline float randomFloat(PPCast::RandomState& randomState) {
        return randomState.random();
    }

    /**
     * @brief Generate a random float in [min, max)
     * 
     * @param min The minimum value
     * @param max The maximum value
     * @param randomState the random state
     * @return a random float in [min, max)
     */
    __host__ __device__ inline float randomFloat(float min, float max, PPCast::RandomState& randomState) {
        return min + (max - min) * randomFloat(randomState);
    }

    /**
     * @brief Generate a random vector where each component is a random number in [-1, 1) 
     * 
     * @tparam L The number of components of the vector
     * @param randomState the random state
     * @return a random float in [min, max)
     */
    template <uint8_t L>
    __host__ __device__ glm::vec<L, float, glm::packed_highp> randomFloatVector(PPCast::RandomState& randomState) {
        glm::vec<L, float, glm::packed_highp> v;
        for (uint8_t i = 0; i < L; ++i) v[i] = randomFloat(-1.f, 1.f, randomState);
        return v;
    }

    /**
     * @brief Generate a random vector within the unit sphere
     * 
     * @tparam L The number of components of the vector 
     * @param randomState the random state
     * @return A random vector within the unit sphere
     */
    template <uint8_t L>
    __host__ __device__ glm::vec<L, float, glm::packed_highp> randomInUnitSphere(PPCast::RandomState& randomState) {
        while (true) {
            const glm::vec<L, float, glm::packed_highp> candidate = randomFloatVector<L>(randomState);
            if (glm::dot(candidate, candidate) < 1.f) return candidate;
        }
    }

    /**
     * @brief Generate a random unit vector
     * 
     * @tparam L The number of components of the vector
     * @param randomState the random state
     * @return A random unit vector
     */
    template <uint8_t L>
    __host__ __device__ glm::vec<L, float, glm::packed_highp> randomUnitVector(PPCast::RandomState& randomState) {
        return glm::normalize(randomInUnitSphere<L>(randomState));
    }

    /**
     * @brief Generate a random unit vector in the same hemisphere as a normal vector
     * 
     * @tparam L The number of components of the vector
     * @param normal The normal vector
     * @param randomState the random state
     * @return a random unit vector in the same hemisphere as the given normal vector
     */
    template <uint8_t L>
    __host__ __device__ glm::vec<L, float, glm::packed_highp> randomOnHemisphere(const glm::vec<L, float, glm::packed_highp>& normal, PPCast::RandomState& randomState) {
        glm::vec<L, float, glm::packed_highp> v = randomUnitVector<L>(randomState);
        return (glm::dot(v, normal) < 0) ? -v : v;
    }
}

#endif