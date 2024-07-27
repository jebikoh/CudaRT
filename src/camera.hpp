#pragma once
#include "rt.hpp"

class Camera {
public:
    jtx::Vec3f origin;
    jtx::Vec3f lowerLeftCorner;
    jtx::Vec3f horizontal;
    jtx::Vec3f vertical;

    __device__ Camera() {
        origin = jtx::Vec3f(0.0f, 0.0f, 0.0f);
        lowerLeftCorner = jtx::Vec3f(-2.0f, -1.0f, -1.0f);
        horizontal = jtx::Vec3f(4.0f, 0.0f, 0.0f);
        vertical = jtx::Vec3f(0.0f, 2.0f, 0.0f);
    }

    __device__ jtx::Rayf getRay(float u, float v) const {
        return {origin, lowerLeftCorner + u * horizontal + v * vertical - origin};
    }
};