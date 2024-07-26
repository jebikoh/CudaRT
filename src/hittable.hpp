#pragma once

#include "rt.hpp"

struct HitRecord {
    jtx::Point3f p;
    jtx::Vec3f normal;
    float t;
    bool frontFace;

    __device__ void setFaceNormal(const jtx::Rayf &r, const jtx::Vec3f &outwardNormal) {
        frontFace = dot(r.dir, outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable {
public:
    __device__ virtual bool hit(const jtx::Rayf &r, const Interval &t, HitRecord &rec) const = 0;
};