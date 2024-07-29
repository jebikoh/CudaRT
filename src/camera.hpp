#pragma once

#include "rt.hpp"

class Camera {
public:
    float aspectRatio;

    int imWidth;
    int imHeight;

    jtx::Point3f center;
    jtx::Point3f upperLeft;
    jtx::Vec3f deltaU;
    jtx::Vec3f deltaV;

    __device__ Camera(int imWidth, int imHeight) : imWidth(imWidth), imHeight(imHeight) {
        aspectRatio = float(imWidth) / float(imHeight);
        center = {0, 0, 0};

        auto focalLength = 1.0f;
        auto viewportHeight = 2.0f;
        auto viewportWidth = viewportHeight * (float(imWidth) / float(imHeight));

        auto u = jtx::Vec3f(viewportWidth, 0, 0);
        auto v = jtx::Vec3f(0, -viewportHeight, 0);

        deltaU = u / float(imWidth);
        deltaV = v / float(imHeight);

        auto viewportUpperLeft = center - jtx::Vec3f{0, 0, focalLength} - u / 2 - v / 2;
        upperLeft = viewportUpperLeft + 0.5 * (deltaU + deltaV);
    }

    __device__ jtx::Rayf getRay(int i, int j, curandState *localRandState) const {
        auto offset = sampleSquare(localRandState);
        auto sample = upperLeft + ((i + offset.x) * deltaU) + ((j + offset.y) * deltaV);
        return {center, sample - center};
    }
};