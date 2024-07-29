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

            if (rec.mat->scatter(r, rec, attenuation, scattered, localRandState)) {
                curAttenuation *= attenuation;
                curRay = scattered;
            } else {
                return {0.0f, 0.0f, 0.0f};
            }
        } else {
            float t = 0.5f * (jtx::normalize(r.dir).y + 1.0f);
            return curAttenuation * jtx::lerp(jtx::Vec3f{1.0f, 1.0f, 1.0f}, jtx::Vec3f{0.5f, 0.7f, 1.0f}, t);
        }
    }
    return {0.0f, 0.0f, 0.0f};
}

__global__ void createWorld(Hittable **d_list,
                            int numHittables,
                            Hittable **d_world,
                            Camera **d_camera,
                            int width,
                            int height
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[1] = new Sphere(jtx::Vec3f(0, -100.5f, -1), 100, new Lambertian(jtx::Vec3f(0.8f, 0.8f, 0.0f)));
        d_list[0] = new Sphere(jtx::Vec3f(0, 0, -1.2), 0.5f, new Lambertian(jtx::Vec3f(0.1f, 0.2f, 0.5f)));
        d_list[3] = new Sphere(jtx::Vec3f(-1, 0, -1), 0.5f, new Dielectric(1.0f / 1.33f));
        d_list[2] = new Sphere(jtx::Vec3f(1, 0, -1), 0.5f, new Metal(jtx::Vec3f(0.8f, 0.6f, 0.2f), 1.0f));
        *d_world = new HittableList(d_list, numHittables);
        *d_camera = new Camera(width, height);
    }
}

__global__ void freeWorld(Hittable **d_list, int numHittables, Hittable **d_world, Camera **d_camera) {
    for (int i = 0; i < numHittables; ++i) {
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
    fb[pixel_index].x = uint8_t(256 * jtx::clamp(::sqrtf(color.r), 0.0f, 0.999f));
    fb[pixel_index].y = uint8_t(256 * jtx::clamp(::sqrtf(color.g), 0.0f, 0.999f));
    fb[pixel_index].z = uint8_t(256 * jtx::clamp(::sqrtf(color.b), 0.0f, 0.999f));
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

    RGB8 *fb;
    CHECK_CUDA(cudaMallocManaged((void **) &fb, fb_size));

    const int numHittables = 4;

    Hittable **d_list;
    CHECK_CUDA(cudaMalloc((void **) &d_list, numHittables * sizeof(Hittable *)));
    Hittable **d_world;
    CHECK_CUDA(cudaMalloc((void **) &d_world, sizeof(Hittable *)));
    curandState *d_randState;
    CHECK_CUDA(cudaMalloc((void **) &d_randState, numPixels * sizeof(curandState)));
    Camera **d_camera;
    CHECK_CUDA(cudaMalloc((void **) &d_camera, sizeof(Camera *)));

    createWorld<<<1, 1>>>(d_list, numHittables, d_world, d_camera, nx, ny);
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
    freeWorld<<<1, 1>>>(d_list, numHittables, d_world, d_camera);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_camera));
    CHECK_CUDA(cudaFree(d_list));
    CHECK_CUDA(cudaFree(d_world));
    CHECK_CUDA(cudaFree(d_randState));
    CHECK_CUDA(cudaFree(fb));
}
