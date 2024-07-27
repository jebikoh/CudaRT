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


using Color = jtx::Vec3f;
using RGB8 = jtx::Vec3<uint8_t>;

void savePNG(const char *path, const RGB8 *fb, int width, int height) {}


#define CHECK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}


__device__ jtx::Vec3f rayColor(const jtx::Rayf &r, Hittable **world) {
    HitRecord rec;
    if ((*world)->hit(r, {0.0f, jtx::INFINITY_F}, rec)) {
        return 0.5f * (rec.normal + 1.0f);
    }
    float t = 0.5f * (jtx::normalize(r.dir).y + 1.0f);
    return jtx::lerp(jtx::Vec3f{1.0f, 1.0f, 1.0f}, jtx::Vec3f{0.5f, 0.7f, 1.0f}, t);
}

__global__ void createWorld(Hittable **d_list, Hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new Sphere(jtx::Point3f{0, 0, -1}, 0.5);
        d_list[1] = new Sphere(jtx::Point3f{0, -100.5, -1}, 100);
        *d_world = new HittableList(d_list, 2);
    }
}

__global__ void freeWorld(Hittable **d_list, Hittable **d_world) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
}

__global__ void renderInit(int maxX, int maxY, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxX) || (j >= maxY)) return;
    int pixelIndex = j * maxX + i;
    curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
}

__global__ void render(RGB8 *fb,
                       int maxX,
                       int maxY,
                       int samplesPerPixel,
                       jtx::Vec3f lowerLeft,
                       jtx::Vec3f horizontal,
                       jtx::Vec3f vertical,
                       jtx::Vec3f origin,
                       Hittable **world,
                       curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxX) || (j >= maxY)) return;
    int pixel_index = j * maxX + i;
    curandState localRandState = randState[pixel_index];
    Color color(0, 0, 0);
    for (int s = 0; s < samplesPerPixel; ++s) {
        float u = (float(i) + curand_uniform(&localRandState)) / float(maxX);
        float v = (float(j) + curand_uniform(&localRandState)) / float(maxY);
        jtx::Rayf r(origin, lowerLeft + u * horizontal + v * vertical);
        color += rayColor(r, world);
    }
    color /= float(samplesPerPixel);
    fb[pixel_index].x = uint8_t(256 * jtx::clamp(color.r, 0.0f, 0.999f));
    fb[pixel_index].y = uint8_t(256 * jtx::clamp(color.g, 0.0f, 0.999f));
    fb[pixel_index].z = uint8_t(256 * jtx::clamp(color.b, 0.0f, 0.999f));
}

int main() {
    const int nx = 1200;
    const int ny = 600;
    const int tx = 8;
    const int ty = 8;
    const int samplesPerPixel = 50;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int numPixels = nx * ny;
    auto fb_size = numPixels * sizeof(RGB8);

    RGB8 *fb;
    CHECK_CUDA(cudaMallocManaged((void **) &fb, fb_size));

    Hittable **d_list;
    CHECK_CUDA(cudaMalloc((void **) &d_list, 2 * sizeof(Hittable *)));
    Hittable **d_world;
    CHECK_CUDA(cudaMalloc((void **) &d_world, sizeof(Hittable *)));
    curandState *d_randState;
    CHECK_CUDA(cudaMalloc((void **) &d_randState, numPixels * sizeof(curandState)));

    createWorld<<<1, 1>>>(d_list, d_world);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    renderInit<<<blocks, threads>>>(nx, ny, d_randState);

    clock_t start, stop;
    start = clock();
    render<<<blocks, threads>>>(fb, nx, ny, samplesPerPixel,
                                jtx::Vec3f(-2.0, -1.0, -1.0),
                                jtx::Vec3f(4.0, 0.0, 0.0),
                                jtx::Vec3f(0.0, 2.0, 0.0),
                                jtx::Vec3f(0.0, 0.0, 0.0),
                                d_world,
                                d_randState);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Time: " << timer_seconds << " seconds\n";

    stbi__flip_vertically_on_write = 1;
    stbi_write_png("output.png", nx, ny, 3, fb, nx * 3);

    CHECK_CUDA(cudaDeviceSynchronize());
    freeWorld<<<1,1>>>(d_list,d_world);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_list));
    CHECK_CUDA(cudaFree(d_world));
    CHECK_CUDA(cudaFree(fb));
}
