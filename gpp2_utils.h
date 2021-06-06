#ifndef GPP2_UTILS_H
#define GPP2_UTILS_H

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

void check(cudaError_t const error) {
    if (error != cudaSuccess)
        throw std::runtime_error{ cudaGetErrorName(error) };
}

namespace gpp2 {
    template <typename T>
    void free(T*& dp) {
        if (dp)
            check(cudaFree(dp));

        dp = nullptr;
    }

    template <typename T> [[nodiscard]]
        T* malloc(std::size_t const size) {
        T* dp{};
        check(cudaMalloc(&dp, sizeof(T) * size));
        return dp;
    }

    template <typename T> [[nodiscard]]
        auto make_unique(std::size_t const size) {
        return std::unique_ptr<T[], decltype (&free<T>)> {malloc<T>(size), free<T>};
    }

}

#endif