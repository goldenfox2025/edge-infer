#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <stdexcept>

namespace op {

/* Owns the process-wide cuBLAS handle. Creation and stream updates are serialized. */
class CUDAResourceManager {
   public:

    static CUDAResourceManager& instance() {
        static CUDAResourceManager instance;
        return instance;
    }

    cublasHandle_t getCublasHandle() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!cublas_handle_initialized_) {
            cublasStatus_t status = cublasCreate(&cublas_handle_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuBLAS handle in CUDAResourceManager");
            }
            cublas_handle_initialized_ = true;
        }
        return cublas_handle_;
    }

    void setCublasStream(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!cublas_handle_initialized_) {
            getCublasHandle();
        }
        cublasStatus_t status = cublasSetStream(cublas_handle_, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set cuBLAS stream in CUDAResourceManager");
        }
    }

    CUDAResourceManager(const CUDAResourceManager&) = delete;
    CUDAResourceManager& operator=(const CUDAResourceManager&) = delete;

   private:

    CUDAResourceManager() : cublas_handle_initialized_(false) {
    }

    ~CUDAResourceManager() {

        if (cublas_handle_initialized_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_initialized_ = false;
        }
    }

    cublasHandle_t cublas_handle_;
    bool cublas_handle_initialized_;

    std::mutex mutex_;
};

}  // namespace op
