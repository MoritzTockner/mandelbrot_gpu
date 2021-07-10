#ifndef MANDELBROT_GPU_H
#define MANDELBROT_GPU_H

#include "bitmap.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <complex>

constexpr static const float g_infinity{ 4 };

template <typename TComplex>
__device__
auto cuda_norm(TComplex const& c) {
    return c.x * c.x + c.y * c.y;
}

template <typename T>
__host__ __device__
auto to_byte(T const value) {
    return static_cast<pfc::byte_t> (value);
}

template <typename TComplex>
__forceinline__ __host__ __device__
pfc::bmp::pixel_t iterate(TComplex const c) {
    std::size_t i{};
    TComplex z{};

#ifdef __CUDA_ARCH__
    while (cuda_norm(z) < g_infinity && ++i < 255) {
        z = cuCaddf(cuCmulf(z, z), c);
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

template <typename TComplex, typename T>
__global__
static void fractal_gpu_kernel(pfc::bmp::pixel_t* const pixels, size_t const width, size_t const height, TComplex const ll, T const d) {
    auto t{ blockIdx.x * blockDim.x + threadIdx.x };

    TComplex const c{ ll.x + d * (t % width), ll.y + d * (t / width) };

    if (t < width * height)
        pixels[t] = iterate(c);

}

//cudaError_t fractal_gpu(dim3 big, dim3 tib, pfc::bmp::pixel_t* const pixels, size_t const width, size_t const height, cuDoubleComplex const ll, cuDoubleComplex const ur);
cudaError_t fractal_gpu(dim3 big, dim3 tib, pfc::bmp::pixel_t* const pixels, size_t const width, size_t const height, cuFloatComplex const ll, cuFloatComplex const ur);

#endif
