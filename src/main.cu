#include <iostream>
#include <time.h>
#include "sphere.hpp"
#include "rt.hpp"
#include "hittable.hpp"
#include "hittablelist.hpp"

using Color = jtx::Vec3f;


#define CHECK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
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

__device__ jtx::Vec3f rayColor(const jtx::Rayf &r, Hittable **world) {
    HitRecord rec;
    if ((*world)->hit(r, {0.0f, jtx::INFINITY_F}, rec)) {
        return 0.5f * (rec.normal + 1.0f);
    }
    float t = 0.5f * (jtx::normalize(r.dir).y + 1.0f);
    return jtx::lerp(jtx::Vec3f{1.0f, 1.0f, 1.0f}, jtx::Vec3f{0.5f, 0.7f, 1.0f}, t);
}

__global__ void render(Color *fb,
                       int max_x,
                       int max_y,
                       jtx::Vec3f lowerLeft,
                       jtx::Vec3f horizontal,
                       jtx::Vec3f vertical,
                       jtx::Vec3f origin,
                       Hittable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    jtx::Rayf r(origin, lowerLeft + u * horizontal + v * vertical);
    fb[pixel_index] = rayColor(r, world);
}

int main() {
    const int nx = 1200;
    const int ny = 600;
    const int tx = 8;
    const int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int numPixels = nx * ny;
    auto fb_size = numPixels * sizeof(Color);

    Color *fb;
    CHECK_CUDA(cudaMallocManaged((void **) &fb, fb_size));

    Hittable **d_list;
    CHECK_CUDA(cudaMalloc((void **) &d_list, 2 * sizeof(Hittable *)));
    Hittable **d_world;
    CHECK_CUDA(cudaMalloc((void **) &d_world, sizeof(Hittable *)));
    createWorld<<<1, 1>>>(d_list, d_world);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, nx, ny,
                                jtx::Vec3f(-2.0, -1.0, -1.0),
                                jtx::Vec3f(4.0, 0.0, 0.0),
                                jtx::Vec3f(0.0, 2.0, 0.0),
                                jtx::Vec3f(0.0, 0.0, 0.0),
                                d_world);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Time: " << timer_seconds << " seconds\n";

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99*fb[pixel_index].x);
            int ig = int(255.99*fb[pixel_index].y);
            int ib = int(255.99*fb[pixel_index].z);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    freeWorld<<<1,1>>>(d_list,d_world);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_list));
    CHECK_CUDA(cudaFree(d_world));
    CHECK_CUDA(cudaFree(fb));
}
