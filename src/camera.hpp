#pragma once

#include "rt.hpp"

class Camera {
public:
    float aspectRatio;

    float vfov;
    jtx::Point3f lookFrom;
    jtx::Point3f lookAt;
    jtx::Vec3f vup;

    float defocusAngle;
    float focusDist;
    jtx::Vec3f defocusDiskU, defocusDiskV;

    int imWidth;
    int imHeight;

    jtx::Point3f center;
    jtx::Point3f upperLeft;
    jtx::Vec3f deltaU;
    jtx::Vec3f deltaV;
    jtx::Vec3f u, v, w;

    __device__ Camera(
            int imWidth,
            int imHeight,
            float vfov = 90.0f,
            jtx::Point3f lookFrom = {0, 0, 0},
            jtx::Point3f lookAt = {0, 0, -1},
            jtx::Vec3f vup = {0, 1, 0},
            float defocusAngle = 0,
            float focusDist = 10
    ) :
            imWidth(imWidth),
            imHeight(imHeight),
            vfov(vfov),
            lookFrom(lookFrom),
            lookAt(lookAt),
            vup(vup),
            defocusAngle(defocusAngle),
            focusDist(focusDist) {
        aspectRatio = float(imWidth) / float(imHeight);
        center = lookFrom;

        float theta = jtx::radians(vfov);
        auto h = tan(theta / 2);
        auto viewportHeight = 2 * h * focusDist;
        auto viewportWidth = viewportHeight * (float(imWidth) / float(imHeight));

        w = (lookFrom - lookAt).normalize();
        u = jtx::cross(vup, w).normalize();
        v = jtx::cross(w, u);

        auto viewportU = viewportWidth * u;
        auto viewportV = viewportHeight * -v;

        deltaU = viewportU / float(imWidth);
        deltaV = viewportV / float(imHeight);

        auto viewportUpperLeft = center - (focusDist * w) - viewportU / 2 - viewportV / 2;
        upperLeft = viewportUpperLeft + 0.5 * (deltaU + deltaV);

        auto defocusRadius = focusDist * tanf(jtx::radians(defocusAngle) / 2);
        defocusDiskU = defocusRadius * u;
        defocusDiskV = defocusRadius * v;
    }

    __device__ jtx::Rayf getRay(int i, int j, curandState *localRandState) const {
        auto offset = sampleSquare(localRandState);
        auto sample = upperLeft + ((i + offset.x) * deltaU) + ((j + offset.y) * deltaV);
        auto rayOrigin = (defocusAngle <= 0) ? center : defocusDiskSample(localRandState);
        auto rayTime = RND;
        return {rayOrigin, sample - rayOrigin, rayTime};
    }

    __device__ jtx::Point3f defocusDiskSample(curandState *localRandState) const {
        auto p = randomInUnitDisk(localRandState);
        return center + p.x * defocusDiskU + p.y * defocusDiskV;
    }
};