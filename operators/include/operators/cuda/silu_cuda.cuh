#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class SiluCUDAOperator : public SiluOperator<T> {
 public:
  SiluCUDAOperator() = default;
  ~SiluCUDAOperator() override = default;

  // 实现CUDA版本的SiLU - 使用一重指针
  void operator()(Tensor<T>* output, Tensor<T>* input,
                  cudaStream_t stream = nullptr) override;

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
