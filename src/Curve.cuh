#ifndef PPCAST_CURVE_H
#define PPCAST_CURVE_H

#include "Common.h"

namespace PPCast {
    /**
     * @brief Types of curves -- this determines the selected interpolation function
     * 
     */
    enum CurveType: uint8_t {
        CONSTANT, // No movement
        LINEAR, // Linear interpolation
    };

    template <typename T>
    class Curve {
    private:
        /// @brief The maximum number of control points (static allocation)
        static constexpr size_t MAX_NUM_CONTROL_POINTS = 2;

        /// @brief The Curve type
        CurveType m_curveType;

        /// @brief The number of control points on the curve (dynamic allocation)
        uint8_t m_numControlPoints;

        /// @brief The curve's control points
        T m_controlPoints[MAX_NUM_CONTROL_POINTS];

        /**
         * @brief Construct a new Curve object
         * 
         * @param curveType The type of curve object
         * @param controlPoints the number of control points
         */
        Curve (CurveType curveType, uint8_t numControlPoints, const std::initializer_list<T> controlPoints)
            : m_curveType(curveType)
            , m_numControlPoints(numControlPoints)
        {
            assert(numControlPoints == controlPoints.size());
            uint8_t i = 0;
            for (const T& c : controlPoints) {
                m_controlPoints[i++] = c;
            }
        }

    public:
        Curve (const Curve& curve) = default;
        Curve (Curve&& curve) = default;
        ~Curve() = default;

        /**
         * @brief Construct a new curve object of the requested type
         * 
         * @tparam curveType the type of curve
         * @param controlPoints the control points for the curve
         * @return A new curve object of the requested type
         */
        template <CurveType curveType>
        static inline Curve makeCurve(const std::initializer_list<T> controlPoints) {
            switch (curveType) {
                case CurveType::CONSTANT: return Curve<T>(curveType, 1, controlPoints);
                case CurveType::LINEAR  : return Curve<T>(curveType, 2, controlPoints);
                default: assert(false); // Invalid curve type
            }
        }

        /**
         * @brief Get the point between the start and end positions at a time t
         * 
         * @param t a time in [0, 1)
         * @return the point between the start and end positions at the given time t
         */
        __host__ __device__ T interpolate(const float t) const;
    };

    struct MotionCurve {
        /// @brief The curve describing the change in position
        Curve<glm::vec3> m_positionCurve;
        
        /// @brief The curve describing the change in rotation
        Curve<glm::mat4> m_rotationCurve;

        /// @brief The curve describing the change in scale
        Curve<glm::vec3> m_scaleCurve;

        /**
         * @brief Get the transformation matrix corresponding to a given time
         * 
         * @param t The time for which to get the transformation matrix
         * @return The transformation matrix corresponding to the given time
         */
        __host__ __device__ glm::mat4 at(const float t) const {
            const glm::mat4 scaleMatrix    = glm::scale(glm::mat4(1.f), m_scaleCurve.interpolate(t));
            const glm::mat4 rotationMatrix = m_rotationCurve.interpolate(t);    
            const glm::mat4 positionMatrix = glm::translate(glm::mat4(1.f), m_positionCurve.interpolate(t));
            return positionMatrix * rotationMatrix * scaleMatrix;
        }
    };

    template <typename T>
    __host__ __device__ static inline T interpolateConstant(const float, const T* c) {
        return c[0];
    }

    template <typename T>
    __host__ __device__ static inline T interpolateLinear(const float t, const T* c) {
        return t * c[0] + (1.f - t) * c[1];
    }

    template <typename T>
    __host__ __device__ T Curve<T>::interpolate(const float t) const {
        switch (m_curveType) {
            case CurveType::LINEAR: return interpolateLinear(t, m_controlPoints);        
            default:                return interpolateConstant(t, m_controlPoints);
        }
    }
}

#endif