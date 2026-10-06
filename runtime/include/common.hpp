#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace edge_infer::detail {

inline void check_cuda(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                             std::to_string(line) + " - " + cudaGetErrorString(status));
}

inline const char* cublas_status_name(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "Unknown cuBLAS status";
  }
}

inline void check_cublas(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLAS error at ") + file + ":" +
                             std::to_string(line) + " - " + cublas_status_name(status) +
                             " (" + std::to_string(static_cast<int>(status)) + ")");
}

}  // namespace edge_infer::detail

// Each call is evaluated once; the helpers cannot shadow the caller's variables.
#define CUDA_CHECK(call) ::edge_infer::detail::check_cuda((call), __FILE__, __LINE__)
#define checkCudaErrors(call) CUDA_CHECK(call)
#define CUBLAS_CHECK(call) ::edge_infer::detail::check_cublas((call), __FILE__, __LINE__)
