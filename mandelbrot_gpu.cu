
#include "mandelbrot_gpu.h"


//cudaError_t fractal_gpu(dim3 big, dim3 tib, pfc::bmp::pixel_t* const pixels, size_t const width, size_t const height, cuDoubleComplex const ll, cuDoubleComplex const ur)
//{
//
//    auto const complex_width{ ur.x - ll.x };
//    auto const d{ complex_width / width };
//
//    fractal_gpu_kernel<<< big, tib >>>(pixels, width, height, ll, d);
//
//    return cudaGetLastError();
//}

cudaError_t fractal_gpu(
    dim3 big, 
    dim3 tib, 
    pfc::bmp::pixel_t* const pixels,
    size_t const width, 
    size_t const height, 
    cuFloatComplex const ll, 
    cuFloatComplex const ur,
    cudaStream_t & stream)
{

    auto const complex_width{ ur.x - ll.x };
    auto const d{ complex_width / width };

    auto const nr_of_pixels = width * height;
    fractal_gpu_kernel << < big, tib, 0, stream >> > (pixels, nr_of_pixels, width, ll, d);

    return cudaGetLastError();
}

cudaError_t fractal_gpu(
    dim3 big,
    dim3 tib,
    pfc::bmp::pixel_t* const pixels,
    size_t const width, 
    size_t const height, 
    cuFloatComplex const ll,
    cuFloatComplex const ur)
{

    auto const complex_width{ ur.x - ll.x };
    auto const d{ complex_width / width };

    auto const nr_of_pixels = width * height;
    fractal_gpu_kernel << < big, tib>> > (pixels, nr_of_pixels, width, ll, d);

    return cudaGetLastError();
}
