#ifndef MANDELBROT_CPU_H
#define MANDELBROT_CPU_H

#include "bitmap.h"
#include <complex>
#include <vector>
#include <future>

template <typename T>
void fractal_multithreaded_kernel(pfc::bitmap& bmp, std::vector<std::size_t> const& line_idxs, std::complex<T> const& ll, T const d) {
    for (auto const& y : line_idxs) {
        for (std::size_t x{ 0 }; x < bmp.width(); ++x) {
            auto const c{ ll + d * std::complex<T>(x, y) };
            bmp.at(x, y) = iterate(c);
        }
    }
}

template <typename T>
pfc::bitmap fractal_multithreaded(std::size_t const nr_tasks, std::size_t const width, std::complex<T> const& ll, std::complex<T> const& ur) {
    auto const complex_width{ ur.real() - ll.real() };
    auto const complex_height{ ur.imag() - ll.imag() };

    pfc::bitmap bmp{ width, static_cast<std::size_t>(complex_height / complex_width * width) };

    auto const d{ complex_width / bmp.width() };

    std::vector<std::future<void>> tasks{};
    std::vector<std::vector<std::size_t>> line_idxs{ nr_tasks, std::vector<std::size_t>{} };

    for (auto y{ 0 }; y < bmp.height(); y++) {
        line_idxs[y % nr_tasks].push_back(y);
    }

    for (auto i{ 0 }; i < nr_tasks; i++) {
        tasks.push_back(std::async(std::launch::async, fractal_multithreaded_kernel<T>, std::ref(bmp), line_idxs[i], ll, d));
    }

    std::for_each(tasks.begin(), tasks.end(), [](auto& t) { t.get(); });

    return bmp;
}

template <typename T>
pfc::bitmap fractal(std::size_t const width, std::complex<T> const& ll, std::complex<T> const& ur) {
    auto const complex_width{ ur.real() - ll.real() };
    auto const complex_height{ ur.imag() - ll.imag() };

    pfc::bitmap bmp{ width, static_cast<std::size_t>(complex_height / complex_width * width) };

    auto const d{ complex_width / bmp.width() };

    for (std::size_t y{ 0 }; y < bmp.height(); ++y) {
        for (std::size_t x{ 0 }; x < bmp.width(); ++x) {
            auto const c{ ll + d * std::complex<T>(x, y) };
            bmp.at(x, y) = iterate(c);
        }
    }

    return bmp;
}

#endif
