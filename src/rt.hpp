#pragma once

#include <jtxlib/math/vec3.hpp>
#include <jtxlib/math/ray.hpp>
#include <jtxlib/math/vecmath.hpp>
#include <curand_kernel.h>
#include "interval.hpp"

#define RANDVEC3 jtx::Vec3f(curand_uniform(localRandState),curand_uniform(localRandState),curand_uniform(localRandState))

__device__ jtx::Vec3f randomInUnitSphere(curandState *localRandState) {
    while (true) {
        auto p = 2.0f * RANDVEC3 - jtx::Vec3f(1, 1, 1);
        if (p.lenSqr() < 1.0f) return p;
    }
}

__device__ jtx::Vec3f randomUnitVector(curandState *localRandState) {
    return jtx::normalize(randomInUnitSphere(localRandState));
}

__device__ jtx::Vec3f randomOnHemisphere(const jtx::Normal3f &normal, curandState *localRandState) {
    jtx::Vec3f unit = randomUnitVector(localRandState);
    if (jtx::dot(unit, normal) > 0.0) return unit;
    return -unit;
}

__device__ jtx::Vec3f randomInUnitDisk(curandState *localRandState) {
    while (true) {
        auto p = jtx::Vec3f(curand_uniform(localRandState), curand_uniform(localRandState), 0) * 2.0f -
                 jtx::Vec3f(1, 1, 0);
        if (p.lenSqr() < 1) return p;
    }
}

__device__ bool nearZero(const jtx::Vec3f &v) {
    // Return true if the vector is close to zero in all dimensions.
    const auto s = 1e-8;
    return (::fabs(v.x) < s) && (::fabs(v.y) < s) && (::fabs(v.z) < s);
}

__device__ jtx::Vec3f sampleSquare(curandState *localRandState) {
    return {curand_uniform(localRandState) - 0.5f, curand_uniform(localRandState) - 0.5f, 0};
}