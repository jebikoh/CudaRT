#pragma once

#include "hittable.hpp"
#include "rt.hpp"


class HittableList : public Hittable {
public:
    Hittable **list;
    int listSize;

    __device__ HittableList() {};
    __device__ HittableList(Hittable **l, int n) : list(l), listSize(n) {}


    __device__ bool hit(const jtx::Rayf &ray, const Interval &t, HitRecord &rec) const override {
        HitRecord tempRec;
        bool hitAnything = false;
        auto closest = t.max;

        for (int i =0; i < listSize; i++) {
            if (list[i]->hit(ray, {t.min, closest}, tempRec)) {
                hitAnything = true;
                closest = tempRec.t;
                rec = tempRec;
            }
        }

        return hitAnything;
    }
};