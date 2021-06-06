
#include "mandelbrot_cpu.h"
#include "mandelbrot_gpu.h"
#include <future>

void fractal_multithreaded_kernel(pfc::bitmap& bmp, std::vector<std::size_t> const& line_idxs, cuDoubleComplex const& ll, cuDoubleComplex c, double const d)
{
    for (auto const& y : line_idxs) {
        for (std::size_t x{ 0 }; x < bmp.width(); ++x) {
            cuDoubleComplex c{ ll.x + d * x, ll.y + d * y };
            bmp.at(x, y) = iterate(c);
        }
    }
}

pfc::bitmap fractal_multithreaded(std::size_t const nr_tasks, std::size_t const width, cuDoubleComplex const& ll, cuDoubleComplex const& ur)
{
    auto const complex_width{ ur.x - ll.x };
    auto const complex_height{ ur.y - ll.y };

    pfc::bitmap bmp{ width, static_cast<std::size_t>(complex_height / complex_width * width) };

    auto c{ ll };
    auto const d{ complex_width / bmp.width() };

    std::vector<std::future<void>> tasks{};
    std::vector<std::vector<std::size_t>> line_idxs{ nr_tasks, std::vector<std::size_t>{} };

    for (auto y{ 0 }; y < bmp.height(); y++) {
        line_idxs[y % nr_tasks].push_back(y);
    }

    for (auto i{ 0 }; i < nr_tasks; i++) {
        tasks.push_back(std::async(std::launch::async, fractal_multithreaded_kernel, std::ref(bmp), line_idxs[i], ll, c, d));
    }

    std::for_each(tasks.begin(), tasks.end(), [](auto& t) { t.get(); });

    return bmp;
}

pfc::bitmap fractal(std::size_t const width, cuDoubleComplex const& ll, cuDoubleComplex const& ur)
{
    auto const complex_width{ ur.x - ll.x };
    auto const complex_height{ ur.y - ll.y };

    pfc::bitmap bmp{ width, static_cast<std::size_t>(complex_height / complex_width * width) };

    auto c{ ll };
    auto const d{ complex_width / bmp.width() };

    for (std::size_t y{ 0 }; y < bmp.height(); ++y) {
        for (std::size_t x{ 0 }; x < bmp.width(); ++x) {
            cuDoubleComplex c{ ll.x + d * x, ll.y + d * y };
            bmp.at(x, y) = iterate(c);
        }
    }

    return bmp;
}





