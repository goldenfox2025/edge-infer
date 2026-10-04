#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "operators/cpu/matmul_cpu.hpp"
#include "operators/cuda/matmul/awq_matmul_cuda.cuh"
#include "operators/cuda/matmul/cublas_matmul_cuda.cuh"
#include "operators/cuda/matmul/cutlass_matmul_cuda.cuh"
#include "operators/cuda/matmul/matmul_cuda.cuh"
#include "operators/operator_base.hpp"
#include "weight_tensor.hpp"

namespace op {

template <typename T>
class MatmulOperatorImpl;

template <typename T>
class MatmulSelector {
   public:
    static MatmulSelector<T>& instance() {
        static MatmulSelector<T> instance;
        return instance;
    }

    void registerCudaImplementations() {

        registerImpl(MatmulType::DEFAULT, OperatorPlatform::CUDA, std::make_shared<MatmulCUDAOperator<T>>());

        registerImpl(MatmulType::CUBLAS, OperatorPlatform::CUDA, std::make_shared<CublasMatmulCUDAOperator<T>>());

        registerImpl(MatmulType::CUTLASS, OperatorPlatform::CUDA, std::make_shared<CutlassMatmulCUDAOperator<T>>());

        registerImpl(MatmulType::AWQ, OperatorPlatform::CUDA, std::make_shared<AwqMatmulCUDAOperator<T>>());
    }

    void registerCpuImplementations() {

        registerImpl(MatmulType::DEFAULT, OperatorPlatform::CPU, std::make_shared<MatmulCPUOperator<T>>());
    }

    void registerImpl(MatmulType type, OperatorPlatform platform, std::shared_ptr<MatmulOperatorImpl<T>> impl) {
        std::string key = getKey(type, platform);
        implementations_[key] = impl;
    }

    std::shared_ptr<MatmulOperatorImpl<T>> getImpl(MatmulType type, OperatorPlatform platform) {
        ensureInitialized(platform);

        std::string key = getKey(type, platform);
        auto it = implementations_.find(key);
        if (it == implementations_.end()) {
            // Fall back to the default implementation when the requested one is absent.
            key = getKey(MatmulType::DEFAULT, platform);
            it = implementations_.find(key);
            if (it == implementations_.end()) {
                return nullptr;
            }
        }
        return it->second;
    }

    std::shared_ptr<MatmulOperatorImpl<T>> selectImpl(const WeightTensor<T>& weight, OperatorPlatform platform) {
        ensureInitialized(platform);

        if (weight.is_quantized()) {
            // Quantized weights select the AWQ implementation.
            return getImpl(MatmulType::AWQ, platform);
        }

        auto cublas_impl = getImpl(MatmulType::CUBLAS, platform);
        if (cublas_impl) {
            return cublas_impl;
        }

        // Fall back to the default implementation if cuBLAS is unavailable.
        return getImpl(MatmulType::DEFAULT, platform);
    }

   private:
    MatmulSelector() = default;
    ~MatmulSelector() = default;

    MatmulSelector(const MatmulSelector&) = delete;
    MatmulSelector& operator=(const MatmulSelector&) = delete;

    void ensureInitialized(OperatorPlatform platform) {
        if (platform == OperatorPlatform::CPU && !cpu_initialized_) {
            registerCpuImplementations();
            cpu_initialized_ = true;
        } else if (platform == OperatorPlatform::CUDA && !cuda_initialized_) {
            registerCudaImplementations();
            cuda_initialized_ = true;
        }
    }

    std::string getKey(MatmulType type, OperatorPlatform platform) {
        return std::to_string(static_cast<int>(type)) + "_" + std::to_string(static_cast<int>(platform));
    }

    std::unordered_map<std::string, std::shared_ptr<MatmulOperatorImpl<T>>> implementations_;

    bool cpu_initialized_ = false;
    bool cuda_initialized_ = false;
};

template class MatmulSelector<float>;
template class MatmulSelector<__nv_bfloat16>;
}  // namespace op
