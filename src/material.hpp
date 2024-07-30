#pragma once

#include "rt.hpp"
#include <jtxlib/math/vecmath.hpp>

class HitRecord;

class Material {
public:
    __device__ virtual bool scatter(
            const jtx::Rayf &ray,
            const HitRecord &rec,
            jtx::Vec3f &attenuation,
            jtx::Rayf &scattered,
            curandState *localRandState) const { return false; }

};

class Lambertian : public Material {
public:
    jtx::Vec3f albedo;

    __device__ Lambertian(const jtx::Vec3f &albedo) : albedo(albedo) {}

    __device__ bool scatter(
            const jtx::Rayf &ray,
            const HitRecord &rec,
            jtx::Vec3f &attenuation,
            jtx::Rayf &scattered,
            curandState *localRandState) const override {
        auto scatterDir = rec.normal + randomUnitVector(localRandState);
        if (nearZero(scatterDir)) scatterDir = rec.normal;
        scattered = jtx::Rayf(rec.p, scatterDir);
        attenuation = albedo;
        return true;
    }
};

class Metal : public Material {
public:
    jtx::Vec3f albedo;
    float fuzz;

    __device__ Metal(const jtx::Vec3f &albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool scatter(
            const jtx::Rayf &ray,
            const HitRecord &rec,
            jtx::Vec3f &attenuation,
            jtx::Rayf &scattered,
            curandState *localRandState) const override {
        jtx::Vec3f reflected = jtx::reflect(ray.dir, rec.normal).normalize();
        reflected += (fuzz * randomUnitVector(localRandState));
        scattered = jtx::Rayf(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.dir, rec.normal) > 0);
    }
};

__device__ float reflectance(float cosine, float refIdx) {
    // Schlick's approximation
    float r0 = (1 - refIdx) / (1 + refIdx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * ::powf((1 - cosine), 5);
}

class Dielectric : public Material {
public:
    float refIdx;

    explicit __device__ Dielectric(float refIdx) : refIdx(refIdx) {}

    __device__ bool scatter(
            const jtx::Rayf &ray,
            const HitRecord &rec,
            jtx::Vec3f &attenuation,
            jtx::Rayf &scattered,
            curandState *localRandState) const override {
        attenuation = jtx::Vec3f(1.0f, 1.0f, 1.0f);
        float ri = rec.frontFace ? (1.0f / refIdx) : refIdx;

        auto unitDir = normalize(ray.dir);
        float cosTheta = fminf(dot(-unitDir, rec.normal), 1.0f);
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

        jtx::Vec3f direction;

        if ((ri * sinTheta > 1.0f) || reflectance(cosTheta, ri) > curand_uniform(localRandState)) {
            direction = jtx::reflect(unitDir, rec.normal);
        } else {
            direction = jtx::refract(unitDir, rec.normal, ri);
        }

        scattered = jtx::Rayf(rec.p, direction);
        return true;
    }
};
