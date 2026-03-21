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

// 前向声明
template <typename T>
class MatmulOperatorImpl;

// MatMul算子选择器 - 单例模式
template <typename T>
class MatmulSelector {
   public:
    static MatmulSelector<T>& instance() {
        static MatmulSelector<T> instance;
        return instance;
    }

    // 注册CUDA实现 - 延迟注册，在需要时调用
    void registerCudaImplementations() {
        // 默认实现
        registerImpl(MatmulType::DEFAULT, OperatorPlatform::CUDA, std::make_shared<MatmulCUDAOperator<T>>());

        // cuBLAS实现
        registerImpl(MatmulType::CUBLAS, OperatorPlatform::CUDA, std::make_shared<CublasMatmulCUDAOperator<T>>());

        // CUTLASS实现
        registerImpl(MatmulType::CUTLASS, OperatorPlatform::CUDA, std::make_shared<CutlassMatmulCUDAOperator<T>>());

        // AWQ实现
        registerImpl(MatmulType::AWQ, OperatorPlatform::CUDA, std::make_shared<AwqMatmulCUDAOperator<T>>());
    }

    // 注册CPU实现 - 延迟注册，在需要时调用
    void registerCpuImplementations() {
        // 默认CPU实现
        registerImpl(MatmulType::DEFAULT, OperatorPlatform::CPU, std::make_shared<MatmulCPUOperator<T>>());
    }

    // 注册实现
    void registerImpl(MatmulType type, OperatorPlatform platform, std::shared_ptr<MatmulOperatorImpl<T>> impl) {
        std::string key = getKey(type, platform);
        implementations_[key] = impl;
    }

    // 获取实现
    std::shared_ptr<MatmulOperatorImpl<T>> getImpl(MatmulType type, OperatorPlatform platform) {
        ensureInitialized(platform);

        std::string key = getKey(type, platform);
        auto it = implementations_.find(key);
        if (it == implementations_.end()) {
            // 如果找不到特定类型的实现，尝试返回默认实现
            key = getKey(MatmulType::DEFAULT, platform);
            it = implementations_.find(key);
            if (it == implementations_.end()) {
                return nullptr;
            }
        }
        return it->second;
    }

    // 根据WeightTensor特性选择合适的实现
    std::shared_ptr<MatmulOperatorImpl<T>> selectImpl(const WeightTensor<T>& weight, OperatorPlatform platform) {
        ensureInitialized(platform);

        if (weight.is_quantized()) {
            // 对于量化权重，使用AWQ实现
            return getImpl(MatmulType::AWQ, platform);
        }

        // 这里可以根据其他特征来选择CUBLAS或CUTLASS
        // 例如，可以根据矩阵大小、批处理等因素

        // 默认使用cuBLAS实现
        auto cublas_impl = getImpl(MatmulType::CUBLAS, platform);
        if (cublas_impl) {
            return cublas_impl;
        }

        // 如果cuBLAS实现不可用，回退到默认实现
        return getImpl(MatmulType::DEFAULT, platform);
    }

   private:
    MatmulSelector() = default;
    ~MatmulSelector() = default;

    // 禁止拷贝和赋值
    MatmulSelector(const MatmulSelector&) = delete;
    MatmulSelector& operator=(const MatmulSelector&) = delete;

    // 确保对应平台的实现已初始化
    void ensureInitialized(OperatorPlatform platform) {
        if (platform == OperatorPlatform::CPU && !cpu_initialized_) {
            registerCpuImplementations();
            cpu_initialized_ = true;
        } else if (platform == OperatorPlatform::CUDA && !cuda_initialized_) {
            registerCudaImplementations();
            cuda_initialized_ = true;
        }
    }

    // 生成算子键
    std::string getKey(MatmulType type, OperatorPlatform platform) {
        return std::to_string(static_cast<int>(type)) + "_" + std::to_string(static_cast<int>(platform));
    }

    // 存储实现的映射表
    std::unordered_map<std::string, std::shared_ptr<MatmulOperatorImpl<T>>> implementations_;

    // 初始化状态标志
    bool cpu_initialized_ = false;
    bool cuda_initialized_ = false;
};

// 显式模板实例化
template class MatmulSelector<float>;
template class MatmulSelector<__nv_bfloat16>;
}  // namespace op
