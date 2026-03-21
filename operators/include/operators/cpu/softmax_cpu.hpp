#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class SoftmaxCPUOperator : public SoftmaxOperator<T> {
 public:
  SoftmaxCPUOperator() = default;
  ~SoftmaxCPUOperator() override = default;

  void operator()(Tensor<T>* output, const Tensor<T>* input, int dim,
                  bool mask = false, int offset = 0,
                  cudaStream_t stream = nullptr) override {
    (void)stream;

    if (!output || !input) {
      throw std::runtime_error("SoftmaxCPUOperator received null tensor");
    }

    const auto& shape = input->sizes();
    if (shape.empty()) {
      throw std::runtime_error("SoftmaxCPUOperator input tensor is empty");
    }

    const size_t rank = shape.size();
    if (dim < 0) {
      dim += static_cast<int>(rank);
    }
    if (dim < 0 || static_cast<size_t>(dim) >= rank) {
      throw std::runtime_error("SoftmaxCPUOperator dimension out of range");
    }

    size_t outer = 1;
    for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
      outer *= shape[i];
    }

    const size_t softmax_dim = shape[dim];
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < rank; ++i) {
      inner *= shape[i];
    }

    const size_t stride = input->strides()[dim];
    const T* input_ptr = input->data_ptr();
    T* output_ptr = output->data_ptr();
    const size_t heads = rank > 1 ? shape[1] : 1;

    for (size_t o = 0; o < outer; ++o) {
      int valid_length = static_cast<int>(softmax_dim);
      if (mask) {
        const size_t query_index = heads == 0 ? 0 : o / heads;
        valid_length = static_cast<int>(offset + query_index + 1);
        if (valid_length > static_cast<int>(softmax_dim)) {
          valid_length = static_cast<int>(softmax_dim);
        }
      }

      for (size_t i = 0; i < inner; ++i) {
        const size_t base_index = o * softmax_dim * inner + i;
        float max_val = -std::numeric_limits<float>::infinity();

        for (size_t j = 0; j < softmax_dim; ++j) {
          const size_t idx = base_index + j * stride;
          const float value =
              (mask && static_cast<int>(j) >= valid_length)
                  ? -std::numeric_limits<float>::infinity()
                  : static_cast<float>(input_ptr[idx]);
          max_val = std::max(max_val, value);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < softmax_dim; ++j) {
          const size_t idx = base_index + j * stride;
          const float value =
              (mask && static_cast<int>(j) >= valid_length)
                  ? -std::numeric_limits<float>::infinity()
                  : static_cast<float>(input_ptr[idx]);
          const float exp_val = std::exp(value - max_val);
          output_ptr[idx] = static_cast<T>(exp_val);
          sum += exp_val;
        }

        if (sum == 0.0f) {
          for (size_t j = 0; j < softmax_dim; ++j) {
            output_ptr[base_index + j * stride] = static_cast<T>(0.0f);
          }
          continue;
        }

        for (size_t j = 0; j < softmax_dim; ++j) {
          const size_t idx = base_index + j * stride;
          output_ptr[idx] = static_cast<T>(static_cast<float>(output_ptr[idx]) / sum);
        }
      }
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
