
#include "mandelbrot_gpu.h"

__device__
auto cuda_norm(cuDoubleComplex const& c) {
    return c.x * c.x + c.y * c.y;
}

__global__ 
static void fractal_gpu_kernel(pfc::bmp::pixel_t* const pixels, size_t const width, size_t const height, cuDoubleComplex const ll, double const d) {
    auto t{ blockIdx.x * blockDim.x + threadIdx.x };

    cuDoubleComplex c{ ll.x + d * (t % width), ll.y + d * (t / width) };

    if (t < width*height)
        pixels[t] = iterate(c);

}

cudaError_t fractal_gpu(dim3 big, dim3 tib, pfc::bmp::pixel_t* const pixels, size_t const width, size_t const height, cuDoubleComplex const ll, cuDoubleComplex const ur)
{

    auto const complex_width{ ur.x - ll.x };
    auto const d{ complex_width / width };

    fractal_gpu_kernel<<< big, tib >>>(pixels, width, height, ll, d);

    return cudaGetLastError();
}
