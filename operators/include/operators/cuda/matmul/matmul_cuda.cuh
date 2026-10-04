#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "operators/operator_base.hpp"
#include "weight_tensor.hpp"

namespace op {

template <typename T>
class MatmulSelector;

template <typename T>
class MatmulCUDAOperator : public MatmulOperatorImpl<T> {
   public:
    MatmulCUDAOperator() = default;
    // Dense submissions use this caller-owned handle. AWQ retains its legacy path.
    explicit MatmulCUDAOperator(cublasHandle_t borrowed_handle);
    ~MatmulCUDAOperator() override = default;

    void operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                    cudaStream_t stream = nullptr) override;

    OperatorPlatform platform() const override {
        return OperatorPlatform::CUDA;
    }

    MatmulType impl_type() const override {
        return MatmulType::DEFAULT;
    }

   private:
    cublasHandle_t borrowed_handle_ = nullptr;
};

}  // namespace op
