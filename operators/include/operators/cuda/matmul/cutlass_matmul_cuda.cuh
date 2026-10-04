#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"
#include "weight_tensor.hpp"

namespace op {

template <typename T>
class CutlassMatmulCUDAOperator : public MatmulOperatorImpl<T> {
   public:
    CutlassMatmulCUDAOperator() = default;
    ~CutlassMatmulCUDAOperator() override = default;

    void operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                    cudaStream_t stream = nullptr) override;

    OperatorPlatform platform() const override {
        return OperatorPlatform::CUDA;
    }

    MatmulType impl_type() const override {
        return MatmulType::CUTLASS;
    }
};

}  // namespace op
