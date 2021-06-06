#ifndef MANDELBROT_GPU_H
#define MANDELBROT_GPU_H

#include "bitmap.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

constexpr double g_infinity{ 4 };

__forceinline__ __host__ __device__
auto cuda_norm(cuDoubleComplex const& c) {
    return c.x * c.x + c.y * c.y;
}

template <typename T>
__host__ __device__
pfc::byte_t to_byte(T const value) {
    return static_cast<pfc::byte_t> (value);
}

__forceinline__ __host__ __device__
pfc::bmp::pixel_t iterate(cuDoubleComplex const c) {
    std::size_t i{};
    cuDoubleComplex z{};

    while (cuda_norm(z) < g_infinity && ++i < 255) {
        z = cuCadd(cuCmul(z, z), c);
    }

    pfc::bmp::pixel_t pixel{};
    pixel.green = to_byte(i);

    return pixel;
}

cudaError_t fractal_gpu(dim3 big, dim3 tib, pfc::bmp::pixel_t* const pixels, size_t const width, size_t const height, cuDoubleComplex const ll, cuDoubleComplex const ur);

#endif
