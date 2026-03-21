#pragma once

#include <algorithm>
#include <stdexcept>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class GatherCPUOperator : public GatherOperator<T> {
 public:
  GatherCPUOperator() = default;
  ~GatherCPUOperator() override = default;

  void operator()(Tensor<T>* output, const Tensor<uint32_t>* input,
                  const Tensor<T>* embedding_table,
                  cudaStream_t stream = nullptr) override {
    (void)stream;

    if (!output || !input || !embedding_table) {
      throw std::runtime_error("GatherCPUOperator received null tensor");
    }

    const auto& table_sizes = embedding_table->sizes();
    if (table_sizes.size() < 2) {
      throw std::runtime_error("GatherCPUOperator requires a 2D embedding table");
    }

    const size_t num_indices = input->numel();
    const size_t embedding_dim = table_sizes[1];

    for (size_t i = 0; i < num_indices; ++i) {
      const uint32_t token_id = input->data_ptr()[i];
      if (token_id >= table_sizes[0]) {
        throw std::runtime_error("GatherCPUOperator token index out of range");
      }

      const T* src = embedding_table->data_ptr() + token_id * embedding_dim;
      T* dst = output->data_ptr() + i * embedding_dim;
      std::copy(src, src + embedding_dim, dst);
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
