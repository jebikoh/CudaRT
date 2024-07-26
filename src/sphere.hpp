#pragma once

#include "hittable.hpp"
#include <jtxlib/math/vec3.hpp>

class Sphere : public Hittable {
public:
    jtx::Point3f center;
    float radius;

    __device__ Sphere() {};
    __device__ Sphere(const jtx::Point3f &center, float radius) : center(center), radius(radius){};

    __device__ bool hit(const jtx::Rayf &ray, const Interval &t, HitRecord &rec) const override {
        jtx::Vec3f oc = center - ray.origin;
        auto a = ray.dir.lenSqr();
        auto h = dot(ray.dir, oc);
        auto c = oc.lenSqr() - radius * radius;
        auto discriminant = h * h - a * c;

        if (discriminant > 0) {
            float root = (h - sqrt(discriminant)) / a;
            if (t.surrounds(root)) {
                rec.t = root;
                rec.p = ray.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
            root = (h + sqrt(discriminant)) / a;
            if (t.surrounds(root)) {
                rec.t = root;
                rec.p = ray.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
        }
        return false;
    }
};