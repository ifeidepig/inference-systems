#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::cerr << __FILE__ << ':' << __LINE__ << ": " \
                  << cudaGetErrorString(error) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

inline bool check_result(const std::vector<float>& result,
                         const std::vector<float>& reference) {
    float max_error = 0;
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i) {
        float error = std::abs(result[i] - reference[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(result[i]) ||
            error > 1e-4f + 1e-4f * std::abs(reference[i])) ok = false;
    }
    std::cout << (ok ? "PASS" : "FAIL") << ", max absolute error = "
              << max_error << '\n';
    return ok;
}
