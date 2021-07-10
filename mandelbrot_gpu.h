#ifndef MANDELBROT_GPU_H
#define MANDELBROT_GPU_H

#include "bitmap.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <complex>

constexpr static const double g_infinity{ 4 };

__device__
auto cuda_norm(cuDoubleComplex const& c);

template <typename T>
__host__ __device__
auto to_byte(T const value) {
    return static_cast<pfc::byte_t> (value);
}

template <typename T>
__forceinline__ __host__ __device__
pfc::bmp::pixel_t iterate(T const c) {
    std::size_t i{};
    T z{};

#ifdef __CUDA_ARCH__
    while (cuda_norm(z) < g_infinity && ++i < 255) {
        z = cuCadd(cuCmul(z, z), c);
    }
#else
    while (std::norm(z) < g_infinity && ++i < 255) {
        z = z * z + c;
    }
#endif

    pfc::bmp::pixel_t pixel{};
    pixel.green = to_byte(i);

    return pixel;
}

cudaError_t fractal_gpu(dim3 big, dim3 tib, pfc::bmp::pixel_t* const pixels, size_t const width, size_t const height, cuDoubleComplex const ll, cuDoubleComplex const ur);

#endif
