#pragma once

#include <algorithm>
#include <stdexcept>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class KvCacheWriteCPUOperator : public KvCacheWriteOperator<T> {
 public:
  KvCacheWriteCPUOperator() = default;
  ~KvCacheWriteCPUOperator() override = default;

  void operator()(const Tensor<T>* src_k, const Tensor<T>* src_v,
                  Tensor<T>* dst_k_cache, Tensor<T>* dst_v_cache,
                  size_t offset, cudaStream_t stream = nullptr) override {
    (void)stream;
    if (!src_k || !src_v || !dst_k_cache || !dst_v_cache) {
      throw std::runtime_error("KvCacheWriteCPUOperator received null tensor");
    }
    if (src_k->sizes() != src_v->sizes()) {
      throw std::runtime_error("KvCacheWriteCPUOperator source shapes mismatch");
    }
    if (src_k->sizes().size() != 3) {
      throw std::runtime_error("KvCacheWriteCPUOperator expects [seq, heads, dim]");
    }

    const size_t seq_len = src_k->sizes()[0];
    const size_t head_size = src_k->sizes()[1] * src_k->sizes()[2];
    if (offset + seq_len > dst_k_cache->sizes()[0] ||
        offset + seq_len > dst_v_cache->sizes()[0]) {
      throw std::runtime_error("KvCacheWriteCPUOperator cache range overflow");
    }

    for (size_t token_idx = 0; token_idx < seq_len; ++token_idx) {
      std::copy(src_k->data_ptr() + token_idx * head_size,
                src_k->data_ptr() + (token_idx + 1) * head_size,
                dst_k_cache->data_ptr() + (offset + token_idx) * head_size);
      std::copy(src_v->data_ptr() + token_idx * head_size,
                src_v->data_ptr() + (token_idx + 1) * head_size,
                dst_v_cache->data_ptr() + (offset + token_idx) * head_size);
    }
  }

  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
