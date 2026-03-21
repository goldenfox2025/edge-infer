#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"
#include "weight_tensor.hpp"

namespace op {

// 前向声明
template <typename T>
class MatmulSelector;

template <typename T>
class MatmulCUDAOperator : public MatmulOperatorImpl<T> {
   public:
    MatmulCUDAOperator() = default;
    ~MatmulCUDAOperator() override = default;

    // 实现CUDA版本的MatMul - 使用智能选择器选择最合适的实现
    void operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                    cudaStream_t stream = nullptr) override;

    // 获取算子平台
    OperatorPlatform platform() const override {
        return OperatorPlatform::CUDA;
    }

    // 获取MatMul算子实现类型
    MatmulType impl_type() const override {
        return MatmulType::DEFAULT;
    }
};

}  // namespace op