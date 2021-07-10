
#include "gpp2_utils.h"
#include "mandelbrot_gpu.h"
#include "mandelbrot_cpu.h"
#include <jobs.h>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <future>
#include <cassert>
#include <algorithm>
#include <sstream>
#include <string>
#include <cuComplex.h>




int main() {
    
    using job_t = pfc::jobs<double>::job_t;

    try {
        int count{};
        check(cudaGetDeviceCount(&count));

        if (count > 0) {
            cudaDeviceProp prop{};
            check(cudaGetDeviceProperties(&prop, 0));

            std::cout << prop.name << " (" << prop.major << "." << prop.minor << ")\n";

            size_t const job_nr = 32;
            auto const width = 2048;
            std::vector<job_t> jobs{};
            for (size_t i{ 0 };  auto const& job : pfc::jobs<double>{ "./jobs/" + pfc::jobs<double>::make_filename(job_nr) }) {
                jobs.push_back(job);
            }
            

            std::vector<pfc::bitmap> bmps_serial{ jobs.size() };
            std::vector<pfc::bitmap> bmps_parallel{ jobs.size() };

            // Calculate fractals serial on CPU
            auto start{ std::chrono::high_resolution_clock::now() };
            for (size_t i{ 0 }; auto const& [ll, ur, cp, wh] : jobs)
                bmps_serial[i++] = fractal(width, ll, ur);
            auto duration_serial{ std::chrono::high_resolution_clock::now() - start };

            std::cout << "Serial fractal duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_serial) << "\n";

            // Calculate fractals in parallel on CPU
            start = std::chrono::high_resolution_clock::now();
            for (size_t i{ 0 }; auto const& [ll, ur, cp, wh] : jobs)
                bmps_parallel[i++] = fractal_multithreaded(std::thread::hardware_concurrency(), width, ll, ur);
            auto duration_parallel{ std::chrono::high_resolution_clock::now() - start };

            std::cout << "Parallel fractal duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_parallel) << "\n";

            // Calculate fractals in parallel on GPU
            std::vector<pfc::bitmap> bmps_parallel_gpu{ jobs.size() };

            int tib{ 32 };
            start = std::chrono::high_resolution_clock::now();
            for (size_t i{ 0 }; auto const& [ll, ur, cp, wh] : jobs) {
                auto const complex_width{ (ur - ll).real() };
                auto const complex_height{ (ur - ll).imag() };
                bmps_parallel_gpu[i].create(width, static_cast<std::size_t>(complex_height / complex_width * width));
                auto const size = bmps_parallel_gpu[i].width() * bmps_parallel_gpu[i].height();
                int big((static_cast<int>(size) + tib - 1) / tib);

                auto dp_pixels{ gpp2::make_unique<pfc::bmp::pixel_t>(size) };

                check(fractal_gpu(big, tib, dp_pixels.get(), bmps_parallel_gpu[i].width(), bmps_parallel_gpu[i].height(), double2{ ll.real(), ll.imag() }, double2{ ur.real(), ur.imag() }));
                check(cudaMemcpy(bmps_parallel_gpu[i].data(), dp_pixels.get(), size * sizeof(pfc::bmp::pixel_t), cudaMemcpyDeviceToHost));
                i++;
            }
            auto duration_parallel_gpu{ std::chrono::high_resolution_clock::now() - start };
            std::cout << "Parallel GPU fractal duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_parallel_gpu) << "\n";

            std::cout << "Speedup parallel CPU: " << duration_serial / duration_parallel << '\n';
            std::cout << "Speedup parallel GPU: " << duration_serial / duration_parallel_gpu << '\n';
            std::cout << "=================================================\n";

            // Compare results
            //for (size_t i{ 0 }; i < bmps_serial.size(); i++) {
            //    for (size_t y{ 0 }; y < bmps_serial[i].height(); y++) {
            //        for (size_t x{ 0 }; x < bmps_serial[i].width(); x++) {
            //            if (bmps_serial[i].at(x, y).green != bmps_parallel[i].at(x, y).green ||
            //                bmps_parallel_gpu[i].at(x, y).green != bmps_parallel[i].at(x, y).green) {
            //                std::cerr << "Results not equal at bitmap nr. " << i << " (" << x << ", " << y << ")\n";
            //                std::cerr << "Serial pixel: " << int{ bmps_serial[i].at(x, y).green } << "\n";
            //                std::cerr << "Parallel pixel: " << int{ bmps_parallel[i].at(x, y).green } << "\n";
            //                std::cerr << "Parallel GPU pixel: " << int{ bmps_parallel_gpu[i].at(x, y).green } << "\n";
            //                //return -1;
            //            }
            //        }
            //    }
            //}

            //std::cout << "Results are equal\n";

            std::string bmp_folder = "bitmaps";
            for (size_t i{ 0 }; auto const& bmp : bmps_serial)
                bmp.to_file(bmp_folder + "/bitmap-serial-" + std::to_string(++i) + ".bmp");
            for (size_t i{ 0 }; auto const& bmp : bmps_parallel)
                bmp.to_file(bmp_folder + "/bitmap-parallel-" + std::to_string(++i) + ".bmp");
            for (size_t i{ 0 }; auto const& bmp : bmps_parallel_gpu)
                bmp.to_file(bmp_folder + "/bitmap-parallel-gpu-" + std::to_string(++i) + ".bmp");
        }
    }
    catch (std::runtime_error const& ex) {
        std::cerr << "Runtime error: " << ex.what() << std::endl;
        return -1;
    }
    catch (std::exception const& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "Unexpected error!" << std::endl;
        return -1;
    }

    check(cudaDeviceReset());

    return 0;
}
