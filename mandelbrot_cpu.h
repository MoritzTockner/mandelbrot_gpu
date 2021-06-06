#ifndef MANDELBROT_CPU_H
#define MANDELBROT_CPU_H

#include "bitmap.h"
#include <cuComplex.h>

#include <vector>

void fractal_multithreaded_kernel(pfc::bitmap& bmp, std::vector<std::size_t> const& line_idxs, cuDoubleComplex const& ll, cuDoubleComplex c, double const d);

pfc::bitmap fractal_multithreaded(std::size_t const nr_tasks, std::size_t const width, cuDoubleComplex const& ll, cuDoubleComplex const& ur);

pfc::bitmap fractal(std::size_t const width, cuDoubleComplex const& ll, cuDoubleComplex const& ur);

#endif
