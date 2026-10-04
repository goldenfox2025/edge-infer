#pragma once

#include <cuda_runtime.h>
#include "operators/operator_base.hpp"
#include "tensor.hpp"

namespace op {

template <typename T>
class SoftmaxCUDAOperator : public SoftmaxOperator<T> {
public:
    virtual ~SoftmaxCUDAOperator() = default;

    void operator()(Tensor<T>* output, const Tensor<T>* input, int dim,
                   bool mask = false, int offset = 0, cudaStream_t stream = nullptr) override;

    OperatorPlatform platform() const override {
        return OperatorPlatform::CUDA;
    }
};

} // namespace op
