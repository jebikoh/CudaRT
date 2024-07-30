#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <time.h>
#include <curand_kernel.h>
#include "sphere.hpp"
#include "rt.hpp"
#include "hittable.hpp"
#include "hittablelist.hpp"
#include "stb_image_write.h"
#include "camera.hpp"
#include "material.hpp"


using Color = jtx::Vec3f;
using RGB8 = jtx::Vec3<uint8_t>;

__device__ int d_nh;

#define CHECK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ jtx::Vec3f rayColor(const jtx::Rayf &r, Hittable **world, curandState *localRandState, int maxDepth = 50) {
    jtx::Rayf curRay = r;
    jtx::Vec3f curAttenuation{1.0f, 1.0f, 1.0f};

    for (int i = 0; i < maxDepth; ++i) {
        HitRecord rec;
        if ((*world)->hit(curRay, {0.001f, jtx::INFINITY_F}, rec)) {
            jtx::Rayf scattered;
            jtx::Vec3f attenuation;

            if (rec.mat->scatter(curRay, rec, attenuation, scattered, localRandState)) {
                curAttenuation *= attenuation;
                curRay = scattered;
            } else {
                return {0.0f, 0.0f, 0.0f};
            }
        } else {
            float t = 0.8f * (jtx::normalize(curRay.dir).y + 1.0f);
            jtx::Vec3f c = jtx::lerp(jtx::Vec3f{1.0f, 1.0f, 1.0f}, jtx::Vec3f{0.5f, 0.7f, 1.0f}, t);
            return curAttenuation * c;
        }
    }
    return {0.0f, 0.0f, 0.0f};
}

__global__ void randInit(curandState *randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, randState);
    }
}

__global__ void createWorld(Hittable **d_list,
                            Hittable **d_world,
                            Camera **d_camera,
                            int width,
                            int height,
                            curandState *randState
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState localRandState = *randState;

        // Ground
        int nh = 0;
        d_list[nh++] = new Sphere(jtx::Vec3f{0, -1000, 0}, 1000, new Lambertian(jtx::Vec3f{0.5, 0.5, 0.5}));
        // Random spheres
        for (int a = -11; a < 11; ++a) {
            for (int b = -11; b < 11; ++b) {
                float chooseMat = curand_uniform(&localRandState);
                jtx::Point3f center(float(a) + 0.9f * curand_uniform(&localRandState), 0.2f,
                                    float(b) + 0.9f * curand_uniform(&localRandState));
                if ((center - jtx::Point3f{4, 0.2f, 0}).len() > 0.9f) {
                    if (chooseMat < 0.8) {
                        // Diffuse
                        auto albedo = randVec3f(&localRandState) * randVec3f(&localRandState);
                        d_list[nh++] = new Sphere(center, 0.2f, new Lambertian(albedo));
                    } else if (chooseMat < 0.95) {
                        // Metal
                        auto albedo = randVec3f(0.5f, 1.0f, &localRandState);
                        auto fuzz = curand_uniform(&localRandState) * 0.5f;
                        d_list[nh++] = new Sphere(center, 0.2f, new Metal(albedo, fuzz));
                    } else {
                        // Glass
                        d_list[nh++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
                    }

                }
            }
        }

        // Three big spheres
        d_list[nh++] = new Sphere(jtx::Vec3f{0, 1, 0}, 1.0f, new Dielectric(1.5f));
        d_list[nh++] = new Sphere(jtx::Vec3f{-4, 1, 0}, 1.0f, new Lambertian(jtx::Vec3f{0.4, 0.2, 0.1}));
        d_list[nh++] = new Sphere(jtx::Vec3f{4, 1, 0}, 1.0f, new Metal(jtx::Vec3f{0.7, 0.6, 0.5}, 0.0f));
        *randState = localRandState;

        d_nh = nh;
        *d_world = new HittableList(d_list, nh);

        float vfov = 20.0f;
        jtx::Point3f lookFrom{13, 2, 3};
        jtx::Point3f lookAt{0, 0, 0};
        jtx::Vec3f vup{0, 1, 0};

        float defocusAngle = 0.6f;
        float focusDist = 10.0f;

        *d_camera = new Camera(width, height, vfov, lookFrom, lookAt, vup, defocusAngle, focusDist);
    }
}

__global__ void freeWorld(Hittable **d_list, Hittable **d_world, Camera **d_camera) {
    for (int i = 0; i < d_nh; ++i) {
        delete *(d_list + i);
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void renderInit(int maxX, int maxY, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= maxX) || (j >= maxY)) return;
    int pixelIndex = j * maxX + i;
    curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
}

__global__ void render(RGB8 *fb,
                       int width,
                       int height,
                       int spp,
                       int maxDepth,
                       Camera **camera,
                       Hittable **world,
                       curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    curandState localRandState = randState[pixel_index];
    Color color(0, 0, 0);
    for (int s = 0; s < spp; ++s) {
        jtx::Rayf r = (*camera)->getRay(i, j, &localRandState);
        color += rayColor(r, world, &localRandState, maxDepth);
    }
    randState[pixel_index] = localRandState;
    color /= float(spp);
    fb[pixel_index].x = uint8_t(255.999 * jtx::clamp(::sqrtf(color.r), 0.0f, 0.999f));
    fb[pixel_index].y = uint8_t(255.999 * jtx::clamp(::sqrtf(color.g), 0.0f, 0.999f));
    fb[pixel_index].z = uint8_t(255.999 * jtx::clamp(::sqrtf(color.b), 0.0f, 0.999f));
}

int main() {
    const int nx = 1200;
    const int ny = 600;
    const int tx = 8;
    const int ty = 8;
    const int spp = 100;
    const int maxDepth = 50;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int numPixels = nx * ny;
    auto fb_size = numPixels * sizeof(RGB8);

    int numHittables = 22 * 22 + 1 + 3;

    RGB8 *fb;
    CHECK_CUDA(cudaMallocManaged((void **) &fb, fb_size));

    Hittable **d_list;
    CHECK_CUDA(cudaMalloc((void **) &d_list, numHittables * sizeof(Hittable *)));
    Hittable **d_world;
    CHECK_CUDA(cudaMalloc((void **) &d_world, sizeof(Hittable *)));
    curandState *d_randState;
    CHECK_CUDA(cudaMalloc((void **) &d_randState, numPixels * sizeof(curandState)));
    curandState *d_worldRandState;
    CHECK_CUDA(cudaMalloc((void **) &d_worldRandState, sizeof(curandState)));
    Camera **d_camera;
    CHECK_CUDA(cudaMalloc((void **) &d_camera, sizeof(Camera *)));

    randInit<<<1, 1>>>(d_worldRandState);
    createWorld<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_worldRandState);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    renderInit<<<blocks, threads>>>(nx, ny, d_randState);

    clock_t start, stop;
    start = clock();
    render<<<blocks, threads>>>(
            fb,
            nx,
            ny,
            spp,
            maxDepth,
            d_camera,
            d_world,
            d_randState);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Time: " << timer_seconds << " seconds\n";

    stbi_write_png("output.png", nx, ny, 3, fb, nx * 3);


    CHECK_CUDA(cudaDeviceSynchronize());
    freeWorld<<<1, 1>>>(d_list, d_world, d_camera);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_camera));
    CHECK_CUDA(cudaFree(d_list));
    CHECK_CUDA(cudaFree(d_world));
    CHECK_CUDA(cudaFree(d_randState));
    CHECK_CUDA(cudaFree(d_worldRandState));
    CHECK_CUDA(cudaFree(fb));
}
