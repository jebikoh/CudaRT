#pragma once

#include "hittable.hpp"
#include <jtxlib/math/vec3.hpp>

class Sphere : public Hittable {
public:
    jtx::Point3f center;
    float radius;
    Material *mat;

    __device__ Sphere(const jtx::Point3f &center, float radius, Material *mat) : center(center), radius(fmaxf(0, radius)), mat(mat){};

    __device__ bool hit(const jtx::Rayf &ray, const Interval &t, HitRecord &rec) const override {
        jtx::Vec3f oc = center - ray.origin;
        auto a = ray.dir.lenSqr();
        auto h = dot(ray.dir, oc);
        auto c = oc.lenSqr() - radius * radius;


        auto discriminant = h * h - a * c;
        if (discriminant < 0) return false;

        auto dsqrt = sqrtf(discriminant);
        auto root = (h - dsqrt) / a;
        if (!t.surrounds(root)) {
            root = (h + dsqrt) / a;
            if (!t.surrounds(root)) {
                return false;
            }
        }

        rec.t = root;
        rec.p = ray.at(root);
        jtx::Vec3f outwardNormal = (rec.p - center) / radius;
        rec.setFaceNormal(ray, outwardNormal);
        rec.mat = mat;

        return true;
    }
};