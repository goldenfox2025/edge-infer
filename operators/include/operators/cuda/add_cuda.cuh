#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class AddCUDAOperator : public AddOperator<T> {
 public:
  AddCUDAOperator() = default;
  ~AddCUDAOperator() override = default;

  // 实现CUDA版本的Add - 使用一重指针
  void operator()(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
                  cudaStream_t stream = nullptr) override;

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
