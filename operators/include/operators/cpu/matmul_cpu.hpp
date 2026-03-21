#pragma once

#include "operators/operator_base.hpp"
#include "weight_tensor.hpp"
namespace op {

template <typename T>
class MatmulCPUOperator : public MatmulOperatorImpl<T> {
   public:
    MatmulCPUOperator() = default;
    ~MatmulCPUOperator() override = default;

    // 实现CPU版本的MatMul
    void operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                    cudaStream_t stream = nullptr) override;

    // 获取算子平台
    OperatorPlatform platform() const override {
        return OperatorPlatform::CPU;
    }

    // 获取MatMul算子实现类型
    MatmulType impl_type() const override {
        return MatmulType::DEFAULT;
    }
};

}  // namespace op