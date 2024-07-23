#include <iostream>

#define CHECK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(float *fb, int max_x, int max_y) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    auto pixelIndex = j * max_x * 3 + i * 3;
    fb[pixelIndex + 0] = float(i) / max_x;
    fb[pixelIndex + 1] = float(j) / max_y;
    fb[pixelIndex + 2] = 0.2;
}

int main() {
    const int width = 256;
    const int height = 256;
    const int tx = 8;
    const int ty = 8;

    int numPixels = width * height;
    size_t fb_size = 3 * numPixels * sizeof(float);

    float *fb;
    CHECK_CUDA(cudaMallocManaged((void **) &fb, fb_size));

    dim3 blocks(width/tx+1, height/ty+1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height-1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j*3*width + i*3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
//            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    CHECK_CUDA(cudaFree(fb));
}
