#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "operators/operator_base.hpp"
#include "weight_tensor.hpp"

namespace op {

template <typename T>
class CublasMatmulCUDAOperator : public MatmulOperatorImpl<T> {
   public:
    CublasMatmulCUDAOperator();
    // The caller owns this handle and must keep it alive for all submissions.
    // Calls sharing a handle must be serialized by the caller.
    explicit CublasMatmulCUDAOperator(cublasHandle_t borrowed_handle);
    ~CublasMatmulCUDAOperator() override;

    void operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                    cudaStream_t stream = nullptr) override;

    OperatorPlatform platform() const override {
        return OperatorPlatform::CUDA;
    }

    MatmulType impl_type() const override {
        return MatmulType::CUBLAS;
    }

   private:
    bool initialized_;

    cublasHandle_t borrowed_handle_ = nullptr;

    // Retained for API compatibility; handle ownership lives in CUDAResourceManager.

    void initialize();

    void destroy();
};

}  // namespace op
