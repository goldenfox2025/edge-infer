#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <stdexcept>

namespace op {

/**
 * @brief CUDA资源管理器类 - 单例模式管理CUDA资源
 *
 * 该类负责管理全局cuBLAS句柄和其他CUDA资源，
 * 确保资源在整个程序生命周期中只被初始化一次，并在程序结束时正确释放。
 */
class CUDAResourceManager {
   public:
    /**
     * @brief 获取单例实例
     * @return CUDAResourceManager的单例实例引用
     */
    static CUDAResourceManager& instance() {
        static CUDAResourceManager instance;
        return instance;
    }

    /**
     * @brief 获取cuBLAS句柄
     * @return cuBLAS句柄
     */
    cublasHandle_t getCublasHandle() {
        std::lock_guard<std::mutex> lock(mutex_);
        // 如果句柄尚未初始化，则初始化它
        if (!cublas_handle_initialized_) {
            cublasStatus_t status = cublasCreate(&cublas_handle_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuBLAS handle in CUDAResourceManager");
            }
            cublas_handle_initialized_ = true;
        }
        return cublas_handle_;
    }

    /**
     * @brief 设置cuBLAS句柄的CUDA流
     * @param stream CUDA流
     */
    void setCublasStream(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!cublas_handle_initialized_) {
            getCublasHandle();  // 确保句柄已初始化
        }
        cublasStatus_t status = cublasSetStream(cublas_handle_, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set cuBLAS stream in CUDAResourceManager");
        }
    }

    // 禁止拷贝构造和赋值操作
    CUDAResourceManager(const CUDAResourceManager&) = delete;
    CUDAResourceManager& operator=(const CUDAResourceManager&) = delete;

   private:
    // 构造函数
    CUDAResourceManager() : cublas_handle_initialized_(false) {
    }

    // 析构函数
    ~CUDAResourceManager() {
        // 释放cuBLAS句柄
        if (cublas_handle_initialized_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_initialized_ = false;
        }
    }

    // cuBLAS句柄
    cublasHandle_t cublas_handle_;
    bool cublas_handle_initialized_;

    // 互斥锁保证线程安全
    std::mutex mutex_;
};

}  // namespace op