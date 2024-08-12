/**
 * This file contains numerical utility functions primarily used by debug asserts
 */

#pragma once

#include <stdexcept>
#include <cmath>

#if defined(CUDA_ENABLED) && defined(__CUDA_ARCH__)
#include <cuda/std/type_traits>

#define JTX_NUM_ONLY(TypeName) template<typename TypeName = T, typename = cuda::std::enable_if_t<cuda::std::is_arithmetic_v<T>>>

#define JTX_NUM_ONLY_T template<typename T, typename = cuda::std::enable_if_t<cuda::std::is_arithmetic_v<T>>>
#define JTX_FP_ONLY_T template<typename T, typename = cuda::std::enable_if_t<cuda::std::is_floating_point_v<T>>>
#define JTX_INT_ONLY_T template<typename T, typename = cuda::std::enable_if_t<cuda::std::is_integral_v<T>>>

#define JTX_ENABLE_FP_BOOL typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<T>, bool>
#define JTX_ENABLE_INT_BOOL typename cuda::std::enable_if_t<cuda::std::is_integral_v<T>, bool>
#define JTX_ENABLE_FP_T typename cuda::std::enable_if_t<cuda::std::is_floating_point_v<T>, T>
#define JTX_ENABLE_INT_T typename cuda::std::enable_if_t<cuda::std::is_integral_v<T>, T>

#else

#define JTX_NUM_ONLY(TypeName) template<typename TypeName = T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>

#define JTX_NUM_ONLY_T template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
#define JTX_FP_ONLY_T template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
#define JTX_INT_ONLY_T template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>

#define JTX_ENABLE_FP_BOOL typename std::enable_if_t<std::is_floating_point_v<T>, bool>
#define JTX_ENABLE_INT_BOOL typename std::enable_if_t<std::is_integral_v<T>, bool>
#define JTX_ENABLE_FP_T typename std::enable_if_t<std::is_floating_point_v<T>, T>
#define JTX_ENABLE_INT_T typename std::enable_if_t<std::is_integral_v<T>, T>

#endif


namespace jtx {
    template<typename T>
    JTX_HOSTDEV JTX_INLINE
    JTX_ENABLE_FP_BOOL
    isNaN(T v) {
#if defined(CUDA_ENABLED) && defined(__CUDA_ARCH__)
        return ::isnan(v);
#else
        return std::isnan(v);
#endif
    }

    template<typename T>
    JTX_HOSTDEV JTX_INLINE
    JTX_ENABLE_INT_BOOL
    isNaN(T v) {
        return false;
    }

    template<typename T>
    JTX_HOSTDEV JTX_INLINE
    JTX_ENABLE_FP_T
    ceil(T v) {
#if defined(CUDA_ENABLED) && defined(__CUDA_ARCH__)
        return ::ceil(v);
#else
        return std::ceil(v);
#endif
    }

    template<typename T>
    JTX_HOSTDEV JTX_INLINE
    JTX_ENABLE_INT_T
    ceil(T v) {
        return v;
    }

    template<typename T>
    JTX_HOSTDEV JTX_INLINE
    JTX_ENABLE_FP_T
    floor(T v) {
#if defined(CUDA_ENABLED) && defined(__CUDA_ARCH__)
        return ::floor(v);
#else
        return std::floor(v);
#endif
    }

    template<typename T>
    JTX_HOSTDEV JTX_INLINE
    JTX_ENABLE_INT_T
    floor(T v) {
        return v;
    }

    template<typename T>
    JTX_HOSTDEV JTX_INLINE T round(T v) {
#if defined(CUDA_ENABLED) && defined(__CUDA_ARCH__)
        return ::round(v);
#else
        return std::round(v);
#endif
    }
} // jtx
