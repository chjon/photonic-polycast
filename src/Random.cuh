#ifndef PPCAST_RANDOM_H
#define PPCAST_RANDOM_H

#include <curand_kernel.h>
#include <random>
#include "Common.h"

namespace PPCast {
    class RandomState {
    private:
        curandState* randomState;
        std::mt19937* randomGenerator;
    public:
        __device__ RandomState(curandState* rs) : randomState(rs), randomGenerator(nullptr) {}
        __host__ RandomState(std::mt19937* rg) : randomState(nullptr), randomGenerator(rg) {}

        __host__ __device__ inline float random() {
            #ifdef __CUDA_ARCH__
            return 1.f - curand_uniform(randomState);
            #else
            std::mt19937& rg = *randomGenerator;
            return static_cast<float>(rg()) / (static_cast<float>(rg.max()) + 1.f);
            #endif
        }
    };
}

// Utils
__host__ __device__ inline float randomFloat(PPCast::RandomState& randomState) {
    return randomState.random();
}

__host__ __device__ inline float randomFloat(float min, float max, PPCast::RandomState& randomState) {
    return min + (max - min) * randomFloat(randomState);
}

template <uint8_t L>
__host__ __device__ glm::vec<L, float, glm::packed_highp> randomFloatVector(PPCast::RandomState& randomState) {
    glm::vec<L, float, glm::packed_highp> v;
    for (uint8_t i = 0; i < L; ++i) v[i] = randomFloat(-1.f, 1.f, randomState);
    return v;
}

template <uint8_t L>
__host__ __device__ glm::vec<L, float, glm::packed_highp> randomInUnitSphere(PPCast::RandomState& randomState) {
    while (true) {
        const glm::vec<L, float, glm::packed_highp> candidate = randomFloatVector<L>(randomState);
        if (glm::dot(candidate, candidate) < 1.f) return candidate;
    }
}

template <uint8_t L>
__host__ __device__ glm::vec<L, float, glm::packed_highp> randomUnitVector(PPCast::RandomState& randomState) {
    return glm::normalize(randomInUnitSphere<L>(randomState));
}

template <uint8_t L>
__host__ __device__ glm::vec<L, float, glm::packed_highp> randomOnHemisphere(const glm::vec<L, float, glm::packed_highp>& normal, PPCast::RandomState& randomState) {
    glm::vec<L, float, glm::packed_highp> v = randomUnitVector<L>(randomState);
    return (glm::dot(v, normal) < 0) ? -v : v;
}

#endif