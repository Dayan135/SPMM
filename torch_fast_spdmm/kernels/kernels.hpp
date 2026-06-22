#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/core/Tensor.h>

#define cuda_check_err(err) _cuda_check_err(err, __FILE__, __LINE__, __func__)

inline void _cuda_check_err(const cudaError_t err, const char* file, const int line, const char* function) {
    if (err != cudaSuccess) {
        auto error = cudaGetErrorString(err);
        std::fprintf(stderr, "Error in %s (%s:%d): %s\n", function, file, line, error);
        throw std::runtime_error(error);
    }
}