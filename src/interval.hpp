#pragma once

#include <jtxlib/math/constants.hpp>

class Interval {
public:
    float min, max;

    __device__ Interval() : min(+jtx::INFINITY_F), max(jtx::NEG_INFINITY_F) {}
    __device__ Interval(float min, float max) : min(min), max(max) {}

    [[nodiscard]] __device__ float size() const { return max - min; }

    [[nodiscard]] __device__ bool contains(float x) const { return x >= min && x <= max; }

    [[nodiscard]] __device__ bool surrounds(float x) const { return x > min && x < max; }

    static const Interval empty, universe;
};

//const Interval Interval::empty = Interval(+jtx::INFINITY_F, jtx::NEG_INFINITY_F);
//const Interval Interval::universe = Interval(jtx::NEG_INFINITY_F, jtx::INFINITY_F);