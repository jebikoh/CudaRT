#pragma once
#include "rt.hpp"

class Camera {
public:
    jtx::Vec3f origin;
    jtx::Vec3f lowerLeftCorner;
    jtx::Vec3f horizontal;
    jtx::Vec3f vertical;
};